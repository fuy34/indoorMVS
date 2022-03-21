# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  Originating Author: Zak Murez (zak.murez.com)

import argparse
import json
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import open3d as o3d
import numpy as np
import torch
import trimesh

from vPlaneRecover.data import SceneDataset, load_info_json
from vPlaneRecover.datasets.scannet import prepare_scannet_scene, prepare_scannet_splits
from vPlaneRecover.datasets.sample import prepare_sample_scene
import vPlaneRecover.transforms as transforms
from vPlaneRecover.tsdf import TSDFFusion, TSDF, coordinates, depth_to_world
from data_prep_util import *

from matplotlib.cm import get_cmap as colormap
from vPlaneRecover.transforms import NYU40_COLORMAP

PI = np.pi
# for 16: control 3 sigma in 1 voxel in one side (3sigma = 16*3),
# for 8 in 2 voxels in total  (3sigma = 5*8)
# for 4 in 5 voxels in total  (3sigma = 11*4)

# STD = {16: 1 , 8: 2, 4: 5 }
ADOPT_THRES = 0.03
NORM_THRES = np.cos(np.deg2rad(30))

def fuse_scene(path, path_meta, scene, voxel_size, trunc_ratio=3, max_depth=3,
               vol_prcnt=.995, vol_margin=1.5, fuse_semseg=False, device=0,
               verbose=2):
    """ Use TSDF fusion with GT depth maps to generate GT TSDFs

    Args:
        path_meta: path to save the TSDFs 
            (we recommend creating a parallel directory structure to save 
            derived data so that we don't modify the original dataset)
        scene: name of scene to process
        voxel_size: voxel size of TSDF
        trunc_ratio: truncation distance in voxel units
        max_depth: mask out large depth values since they are noisy
        vol_prcnt: for computing the bounding volume of the TSDF... ignore outliers
        vol_margin: padding for computing bounding volume of the TSDF
        fuse_semseg: whether to accumulate semseg images for GT semseg
            (prefered method is to not accumulate and insted transfer labels
            from ground truth labeled mesh)
        device: cpu/ which gpu
        verbose: how much logging to print

    Returns:
        writes a TSDF (.npz) file into path_meta/scene

    Notes: we use a conservative value of max_depth=3 to reduce noise in the 
    ground truth. However, this means some distant data is missing which can
    create artifacts. Nevertheless, we found we acheived the best 2d metrics 
    with the less noisy ground truth.
    """

    if verbose>0:
        print('fusing', scene, 'voxel size', voxel_size)

    info_file = os.path.join(path_meta, scene, 'info.json')

    # get gpu device for this worker
    device = torch.device('cuda', device) # gpu for this process

    # get the dataset
    transform = transforms.Compose([transforms.ResizeImage((640,480)),
                                    transforms.ToTensor(),
                                    transforms.InstanceToSemseg('nyu40'),
                                    transforms.IntrinsicsPoseToProjection(),
                                  ])
    frame_types=['depth', 'semseg'] if fuse_semseg else ['depth']
    dataset = SceneDataset(info_file, transform, frame_types)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None,
                                             batch_sampler=None, num_workers=4)

    # find volume bounds and origin by backprojecting depth maps to point clouds
    # use a subset of the frames to save time
    if len(dataset)<=200:
        dataset1 = dataset
    else:
        inds = np.linspace(0,len(dataset)-1,200).astype(np.int) # start:0 end: end, uniformly sample 200
        dataset1 = torch.utils.data.Subset(dataset, inds)
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=None,
                                              batch_sampler=None, num_workers=4)

    pts = []
    for i, frame in enumerate(dataloader1):
        projection = frame['projection'].to(device)
        depth = frame['depth'].to(device)
        depth[depth>max_depth]=0
        pts.append( depth_to_world(projection, depth).view(3,-1).T )
    pts = torch.cat(pts)
    pts = pts[torch.isfinite(pts[:,0])].cpu().numpy()
    # use top and bottom vol_prcnt of points plus vol_margin
    origin = torch.as_tensor(np.quantile(pts, 1-vol_prcnt, axis=0)-vol_margin).float() # choose the 0.005 lowest point along x axis - 1.5 as orginal
    vol_max = torch.as_tensor(np.quantile(pts, vol_prcnt, axis=0)+vol_margin).float() # the most highest point along x axis as end
    vol_dim = ((vol_max-origin)/(float(voxel_size)/100)).int().tolist() # the x, y ,z voxel number


    # initialize tsdf
    tsdf_fusion = TSDFFusion(vol_dim, float(voxel_size)/100, origin,
                             trunc_ratio, device, label=fuse_semseg)

    # integrate frames
    for i, frame in enumerate(dataloader):
        #' keys in frame : 'file_name_image', 'file_name_depth', 'file_name_instance', 'image', 'depth', 'semseg', 'projection'
        if verbose>1 and i%25==0:
            print(scene, 'integrating voxel size', voxel_size, i, len(dataset))

        projection = frame['projection'].to(device)
        image = frame['image'].to(device)
        depth = frame['depth'].to(device)
        semseg = frame['semseg'].to(device) if fuse_semseg else None

        # only use reliable depth
        depth[depth>max_depth]=0

        # this function do Eq.3,4
        tsdf_fusion.integrate(projection, depth, image, semseg)

    # save mesh and tsdf
    file_name_vol = os.path.join(path_meta, scene, 'tsdf_%02d.npz'%voxel_size)
    file_name_mesh = os.path.join(path_meta, scene, 'mesh_%02d.ply'%voxel_size)
    tsdf = tsdf_fusion.get_tsdf()



    tsdf.save(file_name_vol)

    output_meshs = ['color']
    meshes = tsdf.get_mesh(output_meshs)
    for key in meshes:
        if key == 'semseg':
            meshes[key].export(file_name_mesh.replace('.ply', '_semseg.ply'))

        else:
            meshes[key].export(file_name_mesh.replace('.ply', '_%s.ply'%(key)))

    # update info json
    data = load_info_json(info_file)
    data['file_name_vol_%02d'%voxel_size] = file_name_vol
    json.dump(data, open(info_file, 'w'))
    # torch.cuda.empty_cache()

# use labeled mesh to label surface voxels in tsdf
def label_scene(path_meta, scene, voxel_size, dist_thresh=.05, verbose=2, trunc_ratio=3, device=0):
    """ Transfer instance labels from ground truth mesh to TSDF

    For each voxel find the nearest vertex and transfer the label if
    it is close enough to the voxel.

    Args:
        path_meta: path to save the TSDFs 
            (we recommend creating a parallel directory structure to save 
            derived data so that we don't modify the original dataset)
        scene: name of scene to process
        voxel_size: voxel size of TSDF to process
        dist_thresh: beyond this distance labels are not transferd
        verbose: how much logging to print

    Returns:
        Updates the TSDF (.npz) file with the instance volume
    """

    # dist_thresh: beyond this distance to nearest gt mesh vertex, 
    # voxels are not labeled
    # device = torch.device('cuda', 0)
    device = torch.device('cuda', device)

    if verbose>0:
        print('labeling', scene)

    info_file = os.path.join(path_meta, scene, 'info.json')
    data = load_info_json(info_file)

    # each vertex in gt mesh indexs a seg group
    segIndices = json.load(open(data['file_name_seg_indices'], 'r'))['segIndices']

    # maps seg groups to instances
    segGroups = json.load(open(data['file_name_seg_groups'], 'r'))['segGroups']
    mapping = {ind:group['id']+1 for group in segGroups for ind in group['segments']}

    # get per vertex instance ids (0 is unknown, [1,...] are objects)
    n = len(segIndices)
    instance_verts = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        if segIndices[i] in mapping:
            instance_verts[i] = mapping[segIndices[i]]

    # load vertex locations
    mesh = trimesh.load(data['file_name_mesh_gt'], process=False) # if process=True, trimesh will merge vertices and remove NaN values, unacceptable
    verts = mesh.vertices
    normals = mesh.vertex_normals # the fitting mesh has some face deleted, we use original mesh to get init normal
    n_verts = verts.shape[0]

    # the plane information is encoded in color mesh, note we reduplicate the verts to fill the gap in pln mesh, should remove them
    mesh_plane = trimesh.load(data['file_name_plane_mesh'], process=False)
    plane_verts = mesh_plane.vertices[:n_verts]
    assert verts.shape == plane_verts.shape, "plane mesh must have same vertices with orignal mesh" # ensure after merge vertices, they are same
    colors = mesh_plane.visual.vertex_colors.view(np.ndarray)[:n_verts]

    # decode plane info
    plane_param = np.load(data['file_name_plane_param'])
    plane_ins, plane_id2param, new_plane_verts = map_planes(plane_verts, colors,  plane_param, angle_interval=30) # plane_norm, plane_cls,

    # construct kdtree of vertices for fast nn lookup
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(verts)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points  = o3d.utility.Vector3dVector(new_plane_verts)
    kdtree_pln = o3d.geometry.KDTreeFlann(pcd2)

    # load tsdf volume
    tsdf = TSDF.load(data['file_name_vol_%02d'%voxel_size])
    coords_voxel = coordinates(tsdf.tsdf_vol.size(), device=torch.device('cpu'))
    coords_mesh = coords_voxel.type(torch.float) * tsdf.voxel_size + tsdf.origin.T
    mask = tsdf.tsdf_vol.abs().view(-1)<1 #tsdf = max(-1, min(1, sdf/t)),t=0.012 here, if vol == -1 or 1 means invaild

    # transfer vertex instance ids to voxels near surface
    instance_vol = torch.zeros(len(mask), dtype=torch.long)
    normal_vol = get_normal(tsdf.tsdf_vol)
    planeIns_vol = torch.zeros(len(mask), dtype=torch.long).to(device)

    # assign nearest (toward voxel center) point's attribute as voxel value, except for centroid
    for i in mask.nonzero():
        _, inds, dist = kdtree.search_knn_vector_3d(coords_mesh[:,i], 1)
        if dist[0]< tsdf.voxel_size + 0.1:
            instance_vol[i] = instance_verts[inds[0]]
            # planeIns_vol[i] = plane_ins[inds[0]]

        _, inds, dist = kdtree_pln.search_knn_vector_3d(coords_mesh[:, i], 1)
        if dist[0] < (tsdf.voxel_size*3) * trunc_ratio: # large dist and remove later by param filter
            planeIns_vol[i] = plane_ins[inds[0]]

    # remove inaccurate plane Ins
    for i in range(1, plane_id2param.shape[0]):
        cur_plane_param = torch.from_numpy(plane_id2param[i]).float()
        plane_dist = ( cur_plane_param.to(device) @ coords_mesh[:, planeIns_vol == i].float().to(device) - 1).abs() /\
                     cur_plane_param.norm().to(device) / trunc_ratio /tsdf.voxel_size
        planeIns_vol[planeIns_vol == i][plane_dist >=1] = 0

    planeIns_vol = planeIns_vol.cpu()

    # planeCls_vol[0, planeIns_vol > 0] = 1 # first digit control if it is a plane
    tsdf.attribute_vols['instance'] = instance_vol.view(list(tsdf.tsdf_vol.size())) #update previous saving result
    tsdf.attribute_vols['plane_ins'] = planeIns_vol.view(list(tsdf.tsdf_vol.size()))
    tsdf.attribute_vols['plane_norm'] = normal_vol#.view(list(tsdf.tsdf_vol.size()) + [3]).permute([3,0,1,2]) #same as color

    tsdf.save(data['file_name_vol_%02d'%voxel_size],
              addtional_info={'plane_id2param': torch.from_numpy(plane_id2param).float()})

    # viz
    key = 'vol_%02d'%voxel_size
    temp_data = {key:tsdf, 'instances':data['instances'], 'dataset':data['dataset']}
    tsdf = transforms.InstanceToSemseg('nyu40')(temp_data)[key]

    output_meshes = ['semseg', 'plane_ins'] # 'centroid_prob',  'plane_cls',  'normal',
    meshes = tsdf.get_mesh(output_meshes)
    fname = data['file_name_vol_%02d' % voxel_size]

    for key in meshes:
        if key == 'semseg':
            meshes[key].export(fname.replace('tsdf', 'mesh').replace('.npz','_semseg.ply'))

        if key == 'plane_cls':
            meshes[key].export(fname.replace('tsdf', 'mesh').replace('.npz', '_planeCls.ply'))

        if key == 'norm':
            meshes[key].export(fname.replace('tsdf', 'mesh').replace('.npz', '_normal.ply'))

        if key == 'plane_ins':
            meshes[key].export(fname.replace('tsdf', 'mesh').replace('.npz', '_planeIns.ply'))

            _cmap = np.array(NYU40_COLORMAP)
            _mask = planeIns_vol.cpu().view(-1) > 0
            colors = np.zeros([coords_mesh.shape[1], 4])
            n_plane = (planeIns_vol.max() + 1).item()
            if n_plane - 41 > 0:
                cmap = (colormap('jet')(np.linspace(0, 1, n_plane - 41))[:, :3] * 255).astype(np.uint8)
                cmap = cmap[np.random.permutation(n_plane - 41), :]
                plane_color = np.concatenate([_cmap, cmap], axis=0)
            else:
                plane_color = _cmap
            colors[_mask] += np.concatenate([plane_color[planeIns_vol, :],
                                             np.ones([colors.shape[0], 1]) * 255], axis=1)[_mask]
            valid_pnt_mask = colors.sum(axis=1) > 0
            pld = trimesh.points.PointCloud(vertices=coords_mesh.numpy()[:, valid_pnt_mask].T, colors=colors[valid_pnt_mask].clip(0, 255))
            # pld.show() #for debug
            pld.export(fname.replace('tsdf', 'mesh').replace('.npz', '_planeIns_pld.ply'))


def prepare_scannet(path, plane_path, path_meta, i=0, n=1, max_depth=3):
    """ Create all derived data need for the Scannet dataset

    For each scene an info.json file is created containg all meta data required
    by the dataloaders. We also create the ground truth TSDFs by fusing the
    ground truth TSDFs and add semantic labels

    Args:
        path: path to the scannet dataset
        plane_path: path to plane fitting result
        path_meta: path to save all the derived data
            (we recommend creating a parallel directory structure so that 
            we don't modify the original dataset)
        i: process id (used for parallel processing)
            (this process operates on scenes [i::n])
        n: number of processes
        max_depth: mask out large depth values since they are noisy

    Returns:
        Writes files to path_meta
    """
    tmp = {0:0, 1:1, 2:2, 3:3}
    _scenes = []
    valid_scenes = set()
    with open('meta_file/filtered_scenes.txt') as f:
        for line in f:
            valid_scenes.add(line.strip())

    _scenes += sorted([os.path.join('scans', scene)
                      for scene in os.listdir(os.path.join(path, 'scans'))
                      if scene in valid_scenes])

   
    if i==0:
        prepare_scannet_splits(path, path_meta, _scenes)

    with open( 'meta_file/scannet_val.txt') as vf:
        val_list = [os.path.basename(os.path.dirname(line.strip()))for line in vf]
        base_val = set([x[:-3] for x in val_list])

    scene_to_process = []
    for scene in _scenes:
        base_scene = scene[:-3]
        if  os.path.isdir(path_meta) and os.path.isfile(path_meta + '/%s/mesh_04_planeIns.ply'%scene) and \
            os.path.isfile(path_meta + '/%s/mesh_08_planeIns.ply'%scene) and \
                os.path.isfile(path_meta + '/%s/mesh_16_planeIns.ply'%scene): continue

        if base_scene in base_val and base_val[-2:] != '00':
            continue

        scene_to_process.append(scene)


    scenes = scene_to_process[tmp[i]::n]


    for scene in scenes:
        if os.path.isdir(path_meta) and os.path.isfile(path_meta + '/%s/mesh_04_planeIns.ply'%scene) and \
            os.path.isfile(path_meta + '/%s/mesh_08_planeIns.ply'%scene) and \
                os.path.isfile(path_meta + '/%s/mesh_16_planeIns.ply'%scene) :
                continue # if the scene is fully processed, ignore
        print('========processing {} device {}==========='.format(scene, i))
        prepare_scannet_scene(scene, path, plane_path, path_meta, val_list)
        for voxel_size in [4,8,16]:
            fuse_scene(path, path_meta, scene, voxel_size, device=i%8, max_depth=max_depth, trunc_ratio=3)
            if scene.split('/')[0]=='scans':
                label_scene(path_meta, scene, voxel_size,trunc_ratio=3, device=i%8)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuse ground truth tsdf on Scannet')
    parser.add_argument("--path", default= '/data/ScanNet/ScanNet_raw_data/', metavar="DIR", help="path to raw scannet dataset")
    parser.add_argument("--plane_path", default='/data/ScanNet/ScanNet_raw_data/video_plane_fitting/merged_err_thres=0.6/', metavar="DIR",
                        help="path to plane fitting results")
    parser.add_argument("--path_meta", default='/data/ScanNet/planeMVS_data2/', metavar="DIR",   help="path to store processed (derived) dataset")
    parser.add_argument("--dataset", default='scannet', type=str, help="which dataset to prepare")
    parser.add_argument('--i', default=0, type=int,   help='index of part for parallel processing')
    parser.add_argument('--n', default=1, type=int,  help='number of parts to devide data into for parallel processing')
    parser.add_argument('--max_depth', default=3., type=float,  help='mask out large depth values since they are noisy')
    args = parser.parse_args()

    i=args.i
    n=args.n
    assert 0<=i and i<n

    if not os.path.isdir(args.path_meta) :
        os.makedirs(args.path_meta)


    if args.dataset == 'scannet':
        prepare_scannet(
            os.path.join(args.path, 'scannet'),
            args.plane_path,
            os.path.join(args.path_meta, 'scannet'),
            i,
            n,
            args.max_depth
        )

    else:
        raise NotImplementedError('unknown dataset %s'%args.dataset)
