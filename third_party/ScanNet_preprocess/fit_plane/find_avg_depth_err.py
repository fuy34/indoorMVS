import trimesh
import numpy as np
import sys


from matplotlib.cm import get_cmap as colormap
import os
from glob import glob

import cv2
import shutil
import json

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.append('../')
from evaluate import Renderer
from vPlaneRecover.transforms import NYU40_COLORMAP
from vPlaneRecover.datasets.scannet import load_scannet_label_mapping, load_scannet_nyu40_mapping
from fit_plane.util import get_nyu_id2labl

NYU_ID2LAEBL = get_nyu_id2labl()

CLASS_W_LONGSTRIKE = set(['table', 'desk', 'chair', 'window','bookshelf','shelves'])
NONE_PLANE_CLASS = set(['window', 'lamp', 'pillow', 'bag','curtain', 'shower curtain',
                        'toilet', 'bag', 'mirror', 'person', 'clothes', 'sink'])

INVALID_ID = 16777216 // 100 - 1  # (255,255,255) refer to fit_plane code get_gtPlane_segmt.py

PLANE_AREA_THRES = 0.1
PLANE_EDGE_RATIO_THRES = 0.1
PLANE_AREA_RATIO_THRES = 0.5
PLANE_SEC_EDGE_THRES = 0.2

N_DEPTH_THRES = 0.8
DEPTH_ABSDIFF_THRES = 0.1
DEPTH_RELDIFF_THRES = 0.05
N_VERT_THRES = 120 # same as MergedPlaneAreaThreshold = 120 in get_gtPlane_segmt.py

DEPTH_SHIFT = 1000

width, height = 640, 480
zoom_x, zoom_y = width / 1296., height/ 968.

def get_planeColor_map( n_plane):
    # get color
    _cmap = np.array(NYU40_COLORMAP[1:])
    if n_plane - 40 > 0:
        cmap = (colormap('jet')(np.linspace(0, 1, n_plane - 40))[:, :3] * 255).astype(np.uint8)
        cmap = cmap[np.random.permutation(n_plane - 40), :]
        plane_color = np.concatenate([_cmap, cmap], axis=0)
    else:
        plane_color = _cmap

    return plane_color

def color2plane(mesh):
    # convert color w.r.t. plane id
    vert_colors = mesh.visual.vertex_colors.view(np.ndarray)

    chan_0 = vert_colors[:, 2]
    chan_1 = vert_colors[:, 1]
    chan_2 = vert_colors[:, 0]
    plane_id = (chan_2 * 256 ** 2 + chan_1 * 256 + chan_0) // 100 - 1  # there is no (0,0,0) color in fitting mesh
    # mesh.vertex_attributes['plane_id'] = plane_id
    return plane_id

def filter_plane_area_only(mesh):
    unique_id = np.unique(mesh.vertex_attributes['plane_ins'])
    face_color = mesh.visual.face_colors

    # filter them with face area again, we cannot compute bbox, because
    #  hull precision error: Initial simplex is flat, because every piece is flat
    for i, cur_id in enumerate(unique_id):
        if cur_id == INVALID_ID: continue
        # use plane id select vertex
        vert_mask = mesh.vertex_attributes['plane_ins'] == cur_id
        trg_color = mesh.visual.vertex_colors[vert_mask][0]
        face_mask = face_color == trg_color
        sub_mesh = mesh.submesh(np.nonzero(face_mask.all(axis=1)))[0]
        surface_area = sub_mesh.area_faces.sum()
        if surface_area <= PLANE_AREA_THRES:
            mesh.visual.vertex_colors[vert_mask] = np.array([255, 255, 255, 255]).astype(np.uint8)

    return mesh

def filter_plane_instance(mesh, semseg, plane_param, n_plane):
    unique_id = np.unique( mesh.vertex_attributes['plane_ins'])
    face_color = mesh.visual.face_colors
    verts = mesh.vertices
    add_addtion = 0

    for cur_id in unique_id:
        if cur_id == INVALID_ID: continue
        # use plane id select vertex
        vert_mask = mesh.vertex_attributes['plane_ins'] == cur_id
        param = plane_param[cur_id:cur_id+1]

        # debug
        # mesh.visual.vertex_colors[vert_mask] = np.array([255, 0,0,255])

        # load semseg id
        sem_id = np.unique(semseg[vert_mask])
        assert len(sem_id>1)

        #use face color get submesh
        # ensure the plane vert >= 120 as state in fitting
        if vert_mask.sum() < N_VERT_THRES or NYU_ID2LAEBL[sem_id[0]] in NONE_PLANE_CLASS:
            mesh.visual.vertex_colors[vert_mask] = np.array([255, 255, 255, 255]).astype(np.uint8)
            continue


        trg_color = mesh.visual.vertex_colors[vert_mask][0]
        face_mask = face_color == trg_color
        sub_mesh = mesh.submesh(np.nonzero(face_mask.all(axis=1)))[0]
        # NOTE, some verts may do not have a face or have a face with different color with the verts color
        # we ask all verts become white first, and give the largest one meets the requirement their original color
        mesh.visual.vertex_colors[vert_mask] = np.array([255, 255, 255, 255]).astype(np.uint8)



        # if sem_id == 5:
        #     mesh.visual.vertex_colors[vert_mask] = np.array([255, 0, 0, 255]).astype(np.uint8)
        #     mesh.show()
        split_mesh = sub_mesh.split(only_watertight=False)

        # we cannot sort the mesh w.r.t. bbox first, because to compute bbox, there must be at least 4 verts,
        # but some split piece may not have
        verts_masks, surfaces, edge_ratios, sec_edges, area_ratios = [], [], [], [], []
        for i, m in enumerate(split_mesh):
            # https://stackoverflow.com/questions/16210738/implementation-of-numpy-in1d-for-2d-arrays
            vmask = np.in1d(verts.view(dtype='f8,f8,f8').reshape(verts.shape[0]),
                            m.vertices.view(dtype='f8,f8, f8').reshape(m.vertices.shape[0]))

            if vmask.sum() < N_VERT_THRES: continue

            # for each spart
            bbox = m.bounding_box_oriented  # which is oriented bounding box, align with object instead of axis (so it is more tight)
            edges = bbox.primitive.extents
            max_edge, sec_edge = edges[edges.argsort()[-2:][::-1]]
            edge_ratio = sec_edge/max_edge  # to remove single long thin surface

            bbox_area = max_edge * sec_edge
            surface_area = m.area_faces.sum()
            area_ratio = surface_area / bbox_area # to remove L, M, T shape surface

            # if surface_area > max_surface :
            verts_masks.append(vmask)
            surfaces.append(surface_area)
            edge_ratios.append(edge_ratio)
            sec_edges.append(sec_edge)
            area_ratios.append(area_ratio)

        n_assgined = -1
        for  mask, surface, edge_ratio, area_ratio, sec_edge in \
            zip(verts_masks, surfaces, edge_ratios, area_ratios, sec_edges):

            if surface >= PLANE_AREA_THRES and mask.sum() > N_VERT_THRES \
                and sec_edge>=PLANE_SEC_EDGE_THRES and  edge_ratio >= PLANE_EDGE_RATIO_THRES:

                    if NYU_ID2LAEBL[sem_id[0]] in CLASS_W_LONGSTRIKE:
                        if area_ratio >= PLANE_AREA_RATIO_THRES:
                            n_assgined += 1
                    else:
                        n_assgined += 1
                        # mesh.visual.vertex_colors[mask] = trg_color #np.array([255, 0, 255, 255]).astype(np.uint8)

                    # uncomment it if wish to split a large one into pieces to ensure connection
                    # as one plane is splited into 2+, add new param to the result
                    # if n_assgined > 0:
                    #     add_addtion += 1
                    #     more_color = (n_plane + add_addtion) * 100
                    #     new_color = np.array([more_color / (256 * 256), more_color / 256 % 256, more_color % 256, 255])
                    #     plane_param = np.concatenate([plane_param, param])
                    #     mesh.visual.vertex_colors[mask] = new_color

                    # all part meet the standard will be preserved as the original one
                    if n_assgined >= 0:
                        mesh.visual.vertex_colors[mask] = trg_color

        # if n_assgined == -1 and sem_id == 5:
        #     mesh.visual.vertex_colors[vert_mask] = np.array([255, 255, 255, 255]).astype(np.uint8)
        #     print(cur_id, sem_id, ":", surface, PLANE_AREA_THRES, '/', edge_ratio, PLANE_EDGE_RATIO_THRES, '/',
        #     area_ratio, PLANE_AREA_RATIO_THRES, '/', sec_edge, PLANE_SEC_EDGE_THRES)
        #
        # if sem_id == 5:
        #      mesh.show()

    return mesh, plane_param

def load_ply(src, data_pth, scene_name, label2id, scanNet2nyu):
    src_mesh_coarse = trimesh.load(src, process=False)

    # ==============================
    # load instance label agregated idx
    # offer the instance id, and all the over_segmt_pieceID belong to this instance
    # ==============================
    # each vertex in gt mesh indexs a seg group
    segIndices = \
    json.load(open('{}/{}/{}_vh_clean_2.0.010000.segs.json'.format(data_pth, scene_name, scene_name), 'r'))[
        'segIndices']

    # maps seg groups to instances
    segGroups = json.load(open('{}/{}/{}.aggregation.json'.format(data_pth, scene_name, scene_name), 'r'))['segGroups']
    mapping = {ind: group['label'] for group in segGroups for ind in group['segments']}

    # get per vertex instance ids (0 is unknown, [1,...] are objects)
    n = len(segIndices)
    label_verts = np.zeros(n, dtype=np.long)
    for i in range(n):
        if segIndices[i] in mapping:
            label_verts[i] = scanNet2nyu[label2id[mapping[segIndices[i]]]]

    assert len(src_mesh_coarse.vertices) == n

    return src_mesh_coarse, label_verts

def main():
    train_pth = '/data/ScanNet/ScanNet_raw_data/scannet/scannetv2_train.txt'
    data_pth = '/data/ScanNet/ScanNet_raw_data/scannet/scans/'
    frm_err_pth = '/data/ScanNet/ScanNet_raw_data/video_plane_fitting/N_depth_thres=0.8/frm_err_pth/'

    if not os.path.isdir(frm_err_pth):
        os.makedirs((frm_err_pth))


    train_scenes = []
    with open(train_pth) as vf:
        for line in vf:
            tmp = line.strip()
            # if tmp[-2:] == '00':
            train_scenes.append(tmp)
    train_scenes.sort()
    # train_scenes = train_scenes[:3]

    # sum err, num_frame
    total_absDiff = np.zeros([len(train_scenes), 2])
    total_absRel = np.zeros([len(train_scenes),2])
    all_absDiff, all_absRel = [], []
    for k, src in enumerate(train_scenes):

        print('process:', src)

        # read orignal face for depth render, plane fitting mesh have lots of holes between planes
        # Our plane fitting algorithm remove the face if the 3 verts have different plane ins id
        src_orginal_mesh = trimesh.load('{}/{}/{}_vh_clean_2.ply'.format(data_pth, src, src), process=False)

        # ========== render the depth and compare with gt depth ============
        # if inconsistent #frame > N, remove the scene, because either the plane fitting or the gt pose is inaccurate
        # ===================================================================

        # load the gt_depth
        gt_depth_pth = '{}/{}/depth/*.png'.format(data_pth, src)
        depth_list = sorted(glob(gt_depth_pth), key= lambda x: int(os.path.basename(x)[:-4]))

        # load pose path
        gt_pose_pth = '{}/{}/pose/'.format(data_pth, src)

        # load intrinsic
        with open(os.path.join(data_pth, src, '%s.txt' % src)) as info_f:
            info = [line.rstrip().split(' = ') for line in info_f]
            info = {key: value for key, value in info}
            intrinsics = np.array([
                [float(info['fx_color'])*zoom_x, 0., float(info['mx_color'])*zoom_x],
                [0., float(info['fy_color'])*zoom_y, float(info['my_color'])*zoom_y],
                [0., 0., 1.]])

        # init render
        renderer = Renderer()
        mesh_opengl = renderer.mesh_opengl(src_orginal_mesh) # we use the mesh_viz as it has same face as original scannet mesh


        # compare gt depth with render depth
        # as we will use the rendered_depth to generate tsdf_mesh, the mesh will be prune naturally during that process
        sum_absDiff, sum_absRel, n_frm = 0, 0, 0
        scene_stat = {}

        save_pth = frm_err_pth + '/' + src + '.npz'
        if os.path.isfile(save_pth):
            data = np.load(save_pth)
            for key in data.keys():
                sum_absDiff += data[key][0]
                sum_absRel += data[key][1]
                n_frm += 1
                all_absDiff.append(data[key][0])
                all_absRel.append(data[key][1])
        else:

            for iii, depth_pth in enumerate(depth_list):
                frame_name = os.path.basename(depth_pth)[:-4]

                depth_trgt = cv2.imread(depth_pth, -1) / DEPTH_SHIFT
                pose = np.genfromtxt('{}/{}.txt'.format(gt_pose_pth, frame_name))

                if pose[3].sum() != 1:
                    continue

                n_frm += 1

                _, depth_pred = renderer(height, width, intrinsics, pose, mesh_opengl)
                mask = (depth_trgt < 10) * (depth_trgt > 0)
                depth_pred_eval = depth_pred[mask]
                depth_trgt_eval = depth_trgt[mask]

                abs_diff = np.abs(depth_pred_eval - depth_trgt_eval)
                abs_rel = abs_diff / depth_trgt_eval
                sum_absDiff += abs_diff.mean()
                sum_absRel += abs_rel.mean()
                scene_stat[frame_name] = np.array([abs_diff.mean(), abs_rel.mean()])
                all_absDiff.append(abs_diff.mean())
                all_absRel.append(abs_rel.mean())
            np.savez_compressed(save_pth, **scene_stat)

        total_absDiff[k, 0], total_absDiff[k, 1] = sum_absDiff, n_frm
        total_absRel[k, 0], total_absRel[k, 1] = sum_absRel, n_frm
        print('{}: absDiff {} absRel {}'.format(src,  sum_absDiff/ n_frm,  sum_absRel/n_frm))

    np_absDiff_all, np_absRel_all = np.array(all_absDiff), np.array(all_absRel)
    np.save('train_absDiff_stat.npy',total_absDiff)
    np.save('train_absRel_stat.npy',total_absRel)

    print('mean', total_absDiff[:,0].sum()/total_absDiff[:,1].sum(), total_absRel[:,0].sum()/total_absRel[:,1].sum())
    print('std', np_absDiff_all.std(), np_absRel_all.std())
if __name__ == '__main__':
    main()