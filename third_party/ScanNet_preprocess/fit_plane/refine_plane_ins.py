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

sys.path.append(os.path.abspath(__file__))
import pyrender
from util import get_nyu_id2labl, NYU40_COLORMAP,  fitPlane, load_scannet_label_mapping, load_scannet_nyu40_mapping

NYU_ID2LAEBL = get_nyu_id2labl()

CLASS_W_LONGSTRIKE = set(['table', 'desk', 'chair', 'bookshelf','shelves'])
CLASS_LAYOUT = set(['wall', 'floor', 'ceiling']) #'window','door'
NONE_PLANE_CLASS = set([ 'lamp', 'pillow', 'bag','curtain', 'shower curtain',
                        'toilet', 'bag', 'mirror', 'person', 'clothes', 'sink']) #'window',

INVALID_ID = 16777216 // 100 - 1  # (255,255,255) refer to fit_plane code get_gtPlane_segmt.py

PLANE_AREA_THRES = 0.1
PLANE_EDGE_RATIO_THRES = 0.1
PLANE_AREA_RATIO_THRES_1 = 0.5 # for table chair legs
PLANE_AREA_RATIO_THRES_2 = 0.3 # for others T, L shape with slim edge
PLANE_SEC_EDGE_THRES = 0.2

N_DEPTH_THRES = 0.8
DEPTH_ABSDIFF_THRES = 0.20 #0.1  mean in train_set 0.1773466204359862, std 0.3334572764890903
DEPTH_RELDIFF_THRES = 0.07 #0.05 mean in train_set 0.06602691343062499, std  0.1034422194317018
N_VERT_THRES = 120 # same as MergedPlaneAreaThreshold = 120 in get_gtPlane_segmt.py

parallelThreshold = np.cos(np.deg2rad(30))
planeDiffThreshold = 0.05

DEPTH_SHIFT = 1000.

width, height = 640, 480
zoom_x, zoom_y = width / 1296., height/ 968.


class Renderer():
    """OpenGL mesh renderer

    Used to render depthmaps from a mesh for 2d evaluation
    """

    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()



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

def merge_plane_instance(mesh, semseg, plane_param):
    sem_ids = np.unique(semseg)

    # note plane param is stored as n*d
    planeD = np.sqrt(np.sum(plane_param * plane_param, axis=1, keepdims=True))  # d
    planeN = plane_param / planeD
    ret_pln_param = plane_param.copy()

    # diagN = planeN @ planeN.T
    # pairs = np.nonzero(np.triu(diagN, 1) > parallelThreshold)  # mask all lower triangle and diag
    verts = mesh.vertices.view(np.ndarray).copy()

    for sem_id in sem_ids:
        # only merge layout planes
        if NYU_ID2LAEBL[sem_id] not in CLASS_LAYOUT:
            continue

        vert_mask = semseg == sem_id
        plane_ins_arr = mesh.vertex_attributes['plane_ins']
        selected_plane_ins =  np.unique(plane_ins_arr[vert_mask])

        # ====debug===
        # tmp = mesh.visual.vertex_colors[vert_mask][0]
        # mesh.visual.vertex_colors[vert_mask] = np.array([255, 0, 0, 255]).astype(np.uint8)
        # mesh.show()
        # mesh.visual.vertex_colors[vert_mask] = tmp
        # =======

        # try to merge if they are close
        n_ins = len(selected_plane_ins)
        for i in range(n_ins):
            cur_pln_id = selected_plane_ins[i]
            cur_mask = plane_ins_arr==cur_pln_id
            cur_verts = verts[cur_mask]
            if cur_pln_id == INVALID_ID or cur_mask.sum() == 0 or np.abs(ret_pln_param[cur_pln_id]).sum()==0:
                # invalid plane, or plane has been merged
                continue

            for j in range(i+1, n_ins):
                nxt_pln_id = selected_plane_ins[j]
                nxt_mask = plane_ins_arr==nxt_pln_id
                nxt_verts = verts[nxt_mask]

                if nxt_pln_id == INVALID_ID or nxt_mask.sum() == 0 :
                    # invalid plane, or plane has been merged
                    continue

                planeNorm = np.linalg.norm(planeN[cur_pln_id])
                assert planeNorm > 0
                diff = np.abs(np.matmul(nxt_verts, planeN[cur_pln_id]) - planeD[cur_pln_id]) / planeNorm
                if diff.mean() < planeDiffThreshold:
                    # ========= debug
                    # tmp = mesh.visual.vertex_colors[cur_mask][0]
                    # mesh.visual.vertex_colors[cur_mask] = np.array([255, 0, 0, 255]).astype(np.uint8)
                    # mesh.visual.vertex_colors[nxt_mask] = np.array([255, 255, 0, 255]).astype(np.uint8)
                    # mesh.show()
                    # mesh.visual.vertex_colors[cur_mask] = tmp
                    # ==========

                    # means cur plane param is suitable for nxt plane as well, merge them
                    all_verts = np.concatenate([cur_verts, nxt_verts], axis=0)
                    all_norms = np.stack([planeN[cur_pln_id], planeN[nxt_pln_id]], axis=0)

                    # try 2 fitting w/ w/o norm and pick better one
                    new_param1 = fitPlane(all_verts, all_norms)
                    new_D1 = 1 / np.sqrt((new_param1**2).sum())
                    new_N1 = new_param1 * new_D1
                    err1 = np.abs(np.matmul(all_verts, new_N1) - new_D1) / np.linalg.norm(new_N1)

                    new_param2 = fitPlane(all_verts)
                    new_D2= 1/ np.linalg.norm(new_param2)
                    new_N2 = new_param2 * new_D2
                    err2 = np.abs(np.matmul(all_verts, new_N2) - new_D2) / np.linalg.norm(new_N2)

                    if err1.mean() < err2.mean():
                        new_param = new_N1 * new_D1 #new_param1 * (new_D1**2)
                    else:
                        new_param =  new_N2 * new_D2

                    print(cur_pln_id, nxt_pln_id)
                    print('old', planeD[cur_pln_id], planeN[cur_pln_id])
                    print( np.sqrt(np.sum(new_param * new_param, keepdims=True)),
                           new_param/ np.sqrt(np.sum(new_param * new_param, keepdims=True)))

                    # update color and param
                    mesh.visual.vertex_colors[nxt_mask] = mesh.visual.vertex_colors[cur_mask][0]
                    ret_pln_param[cur_pln_id] = new_param
                    ret_pln_param[nxt_pln_id] = 0
                    plane_ins_arr[nxt_mask] = cur_pln_id

    return mesh, ret_pln_param

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
        assert len(sem_id)==1

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
                        if area_ratio >= PLANE_AREA_RATIO_THRES_1:
                            n_assgined += 1
                    else:
                        if area_ratio >= PLANE_AREA_RATIO_THRES_2:
                            n_assgined += 1
                        # mesh.visual.vertex_colors[mask] = trg_color #np.array([255, 0, 255, 255]).astype(np.uint8)

                    # ======================
                    # uncomment it if wish to split a large one into pieces to ensure connection
                    # as one plane is splited into 2+, add new param to the result
                    # if n_assgined > 0:
                    #     add_addtion += 1
                    #     more_color = (n_plane + add_addtion) * 100
                    #     new_color = np.array([more_color / (256 * 256), more_color / 256 % 256, more_color % 256, 255])
                    #     plane_param = np.concatenate([plane_param, param])
                    #     mesh.visual.vertex_colors[mask] = new_color
                    # =======================

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

if __name__ == '__main__':
    val_pth = '/data/ScanNet/ScanNet_raw_data/scannet/scannetv2_val.txt'
    data_pth = '/data/ScanNet/ScanNet_raw_data/scannet/scans/' #'/data/ScanNet/ScanNet_raw_data_all/scans' #

    src_pth = '/data/ScanNet/ScanNet_raw_data/video_plane_fitting/mesh/'
    src_param_pth ='/data/ScanNet/ScanNet_raw_data/video_plane_fitting/merged_no_planeProj_thres=0.8/plane_param/'
    # src_depth_err = '/data/ScanNet/ScanNet_raw_data/video_plane_fitting/merged_no_planeProj_thres=0.8/frm_err_pth'

    trg_filter_pth =  '/data/ScanNet/ScanNet_raw_data/video_plane_fitting/merged_err_thres={}/filter_mesh_no_gap/'.format(N_DEPTH_THRES)
    trg_viz_pth = '/data/ScanNet/ScanNet_raw_data/video_plane_fitting/merged_err_thres={}/recolor_mesh_no_gap/'.format(N_DEPTH_THRES)
    # trg_dump_depth ='/data/ScanNet/ScanNet_raw_data/video_plane_fitting/merged_no_planeProj_thres=0.8/rendered_depth/'
    trg_param_pth = '/data/ScanNet/ScanNet_raw_data/video_plane_fitting/merged_err_thres={}/refined_pln_param/'.format(N_DEPTH_THRES)

    scanet_label2id = load_scannet_label_mapping(
        '/data/ScanNet/ScanNet_raw_data/scannet/scannetv2-labels.combined.tsv')
    scannet2nyu = load_scannet_nyu40_mapping(
        '/data/ScanNet/ScanNet_raw_data/scannet/scannetv2-labels.combined.tsv')

    if not os.path.isdir(trg_filter_pth):
        os.makedirs((trg_filter_pth))

    if not os.path.isdir(trg_viz_pth):
        os.makedirs(trg_viz_pth)

    # if not os.path.isdir(trg_dump_depth):
    #     os.makedirs(trg_dump_depth)

    if not os.path.isdir(trg_param_pth):
        os.makedirs(trg_param_pth)

    val_scenes = []
    # val_scenes_all =[]
    with open(val_pth) as vf:
        for line in vf:
            tmp = line.strip()
            # if tmp[-2:] == '00':
            val_scenes.append(tmp)

    processed_scenes = glob(trg_filter_pth+'/*.ply')
    processed_scene = [os.path.basename(x)[:-11] for x in processed_scenes]
    src_list = sorted(glob(src_pth + '/*.ply'))


    filter_rec = []
    for src in src_list:
        invalid_pose = []
        inaccurate_pose = []
        scene_name = os.path.basename(src)[:-11]
        print('process:', scene_name)
        # if scene_name in processed_scene: continue
        if scene_name in processed_scene: continue
        if not os.path.isdir(os.path.join(data_pth, scene_name)):continue
        # if scene_name != 'scene0011_00': continue

        # ========= read ply =============
        src_mesh_coarse, src_semseg = load_ply(src, data_pth, scene_name, scanet_label2id, scannet2nyu)
        src_mesh_coarse.vertex_attributes['plane_ins'] = color2plane(src_mesh_coarse)

        # ========= load param ==============
        param_pth = os.path.join(src_param_pth, scene_name + '_planes.npy')
        plane_param = np.load(param_pth)
        n_plane_ever_exist = len(plane_param)

        # =================== compare gt depth with render depth to check pose and depth=====================

        renderer = Renderer()
        mesh_opengl = renderer.mesh_opengl(src_mesh_coarse)  # we use the mesh_viz as it has same face as original scannet mesh
        n_diff = 0
        b_ignore = False

        # we use rendered depth to check if the scene is valid, we already record the valid scenes in meta_file, valid_scenes,
        # so we do not need to record render_depth anymore. But feel free to uncomment the corresponding lines if you wish to
        # filter the scenes in a different way (e.g., difference depth error threshold)
        # scene_depth_dump_path = trg_dump_depth + '/{}/render_depth'.format(scene_name)

        # load the gt_depth
        gt_depth_pth = '{}/{}/depth/*.png'.format(data_pth, scene_name)
        depth_list = sorted(glob(gt_depth_pth), key=lambda x: int(os.path.basename(x)[:-4]))
        cur_depth_thres_num = len(depth_list) * N_DEPTH_THRES


        # load pose path
        gt_pose_pth = '{}/{}/pose/'.format(data_pth, scene_name)

        # load intrinsic
        with open(os.path.join(data_pth, scene_name, '%s.txt' % scene_name)) as info_f:
            info = [line.rstrip().split(' = ') for line in info_f]
            info = {key: value for key, value in info}
            intrinsics = np.array([
                [float(info['fx_color']) * zoom_x, 0., float(info['mx_color']) * zoom_x],
                [0., float(info['fy_color']) * zoom_y, float(info['my_color']) * zoom_y],
                [0., 0., 1.]])


        if scene_name not in val_scenes:
            # only filter the training set
            for iii, depth_pth in enumerate(depth_list):
                frame_name = os.path.basename(depth_pth)[:-4]

                depth_trgt = cv2.imread(depth_pth, -1) / DEPTH_SHIFT
                pose = np.genfromtxt('{}/{}.txt'.format(gt_pose_pth, frame_name))

                if pose[3].sum() != 1:
                    invalid_pose.append('{}\n'.format( frame_name))
                    inaccurate_pose.append('{}\n'.format( frame_name))
                    n_diff += 1
                    continue

                _, depth_pred = renderer(height, width, intrinsics, pose, mesh_opengl)
                mask = (depth_trgt < 10) * (depth_trgt > 0)
                depth_pred_eval = depth_pred[mask]
                depth_trgt_eval = depth_trgt[mask]

                abs_diff = np.abs(depth_pred_eval - depth_trgt_eval)
                abs_rel = abs_diff / depth_trgt_eval

                if abs_diff.mean() > DEPTH_ABSDIFF_THRES and abs_rel.mean() > DEPTH_RELDIFF_THRES:
                    n_diff += 1
                    inaccurate_pose.append('{}\n'.format(frame_name))
                    sys.stdout.write(
                        '\rid {}/{}, {}/{}: abs_diff {:.2f}, thres {:.2f}, abs_rel {:.2f}, thres {:.2f}'.format(
                            iii, len(depth_list), n_diff, int(cur_depth_thres_num), abs_diff.mean(), DEPTH_ABSDIFF_THRES,
                            abs_rel.mean(), DEPTH_RELDIFF_THRES))
                    sys.stdout.flush()
                else:
                    sys.stdout.write('\rpose check {}/{}'.format(  iii, len(depth_list)))
                    sys.stdout.flush()
                # else:
                #     cv2.imwrite(trg_dump_depth + '/{}/render_depth/{}.png'.format(scene_name,frame_name), (depth_pred*DEPTH_SHIFT).astype(np.uint16))

                if n_diff > cur_depth_thres_num:
                    print("{}/{} frames have different GT and rendered depth, remove the scene".format(n_diff,
                                                                                                       len(depth_list)))
                    b_ignore = True
                    break

            with open( '{}/{}/invalid_pose.txt'.format(data_pth, scene_name), 'w') as f:
                for line in invalid_pose:
                    f.write(line)

            with open('{}/{}/inaccurate_pose.txt'.format(data_pth, scene_name), 'w') as f:
                for line in inaccurate_pose:
                    f.write(line)

        # No need to continue if the pose of the whole scene is bad
        if b_ignore:
            continue

        #===  we first filter the plane instance (optionally include split not connected pieces into multiple instance)
        src_mesh, plane_param = filter_plane_instance(src_mesh_coarse, src_semseg, plane_param, n_plane_ever_exist)

        src_mesh.vertex_attributes['plane_ins'] = color2plane(src_mesh)
        src_mesh, ret_plane_param = merge_plane_instance(src_mesh, src_semseg, plane_param)
        verts, faces = src_mesh.vertices.view(np.ndarray).copy(), src_mesh.faces

        # convert plane param note we save plane_param as n*d (consistent with planeRCNN)
        # if you wish to change to n, d version, use the following code
        # planeD = np.linalg.norm(ret_plane_param, axis=1, keepdims=True)  # d
        # planeN = ret_plane_param / planeD

        # read orignal face for depth render, plane fitting mesh have lots of holes between planes
        # Our plane fitting algorithm remove the face if the 3 verts have different plane ins id
        src_orginal_mesh = trimesh.load('{}/{}/{}_vh_clean_2.ply'.format(data_pth, scene_name, scene_name), process=False)
        original_faces =  src_orginal_mesh.faces

        # ======== duplicated verts to fill in the gap =========
        # deleted faces in plane instance fitting
        face_complement_mask = np.isin(original_faces.view(dtype='int64,int64,int64').reshape(original_faces.shape[0]),
                        faces.view(dtype='int64, int64, int64').reshape(faces.shape[0]))
        face_complement = original_faces[~face_complement_mask]

        fh, fw = face_complement.shape
        face_complement_src = face_complement.reshape([-1])

        # copy verts
        complement_vert_id = np.unique(face_complement)
        duplicate_verts = verts[complement_vert_id]

        duplicate_faces_id = np.arange(0, complement_vert_id.shape[0]) + verts.shape[0]
        map_dict = {x:y for x, y in zip(complement_vert_id, duplicate_faces_id)}
        # face_complement_tgt[map_dict.keys()] = map_dict.values()
        def mp(vec):
            return map_dict[vec]
        mp_func = np.vectorize(mp)
        face_complement_tgt = mp_func(face_complement_src)
        face_complement_tgt = face_complement_tgt.reshape([fh, fw])

        # convert color w.r.t. plane id
        vert_colors = src_mesh.visual.vertex_colors.view(np.ndarray).copy()

        plane_id = color2plane(src_mesh)
        unique_id = np.unique(plane_id)

        # intial new color for viz
        new_colors = np.zeros_like(vert_colors[:, :3]) # for vizualization
        plane_color_map = get_planeColor_map(len(unique_id) - 1)

        # ========= refine the mesh by merge and area filter ==========
        plane_verts = np.concatenate([verts, duplicate_verts], axis=0)
        plane_faces = np.concatenate([faces, face_complement_tgt], axis=0)
        plane_color = np.concatenate([vert_colors, np.ones([duplicate_verts.shape[0], 4])*255]).astype(np.uint8)

        for k, id in enumerate(unique_id):
            if id == INVALID_ID: continue

            mask = plane_id == id
            new_colors[mask] = plane_color_map[k]


        mesh_viz = trimesh.Trimesh(vertices=verts, faces=original_faces, vertex_colors=new_colors, process=False)
        mesh = trimesh.Trimesh(vertices=plane_verts, faces=plane_faces, vertex_colors=plane_color, process=False) #faces
        mesh.vertex_attributes['plane_ins'] = color2plane(mesh) # get id
        mesh = filter_plane_area_only(mesh)
        unique_id = np.unique(color2plane(mesh))

        mesh_viz.export(trg_viz_pth + '/' + scene_name + '_planes.ply')
        mesh.vertex_attributes = {}
        mesh.export(trg_filter_pth + '/' + scene_name + '_planes.ply')
        np.save(trg_param_pth + '/{}_planes.npy'.format(scene_name), ret_plane_param)
