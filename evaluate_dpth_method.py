
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
import open3d as o3d
import numpy as np
import pyrender
import torch
import trimesh

from vPlaneRecover.data import SceneDataset, parse_splits_list
from vPlaneRecover.evaluation import eval_tsdf, eval_mesh, eval_depth, project_to_mesh
import vPlaneRecover.transforms as transforms
from vPlaneRecover.tsdf import TSDF, TSDFFusion
from visualize_metrics import visualize
from collections import defaultdict
from glob import glob
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Renderer():
    """OpenGL mesh renderer 
    
    Used to render depthmaps from a mesh for 2d evaluation
    """
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        #self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)#, self.render_flags) 

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R =  np.array([[1, 0, 0],
                       [0, c, -s],
                       [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose@axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()
        

def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, coord_2d):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])

    # x_ref = coord_2d[:1]

    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        coord_2d * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(np.linalg.inv(extrinsics_src), (extrinsics_ref)), #scannet pose * cam = world
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR) # like grid_sample
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(np.linalg.inv(extrinsics_ref), extrinsics_src),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src,
                                geo_pixel_thres, geo_depth_thres, coord_2d):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src, coord_2d)
    # print(depth_ref.shape)
    # print(depth_reprojected.shape)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    # depth_ref = np.squeeze(depth_ref, 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < geo_pixel_thres, relative_depth_diff < geo_depth_thres)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def process(info_file, tsdf_pth, gt_pth, dpth_pth, save_path, total_scenes_index, total_scenes_count):

    # gt loader
    width, height = 640, 480
    transform = transforms.Compose([
        transforms.ResizeImage((width,height)),
        transforms.ToTensor(),
    ])
    dataset = SceneDataset(info_file, transform, frame_types=['depth'])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=None,
    #                                          batch_sampler=None, num_workers=2)
    scene = dataset.info['scene']

    # get info about tsdf
    file_tsdf_pred = os.path.join(tsdf_pth, scene, 'tsdf_08.npz')
    temp = TSDF.load(file_tsdf_pred)
    voxel_size = int(temp.voxel_size*100)

    # re-fuse to remove hole filling since filled holes are penalized in
    # mesh metrics, but do nothing if the hole is not caused by visiablity
    vol_dim = list(temp.tsdf_vol.shape)
    origin = temp.origin
    tsdf_fusion = TSDFFusion(vol_dim, float(voxel_size) / 100, origin, color=False)
    device = tsdf_fusion.device

    pose_list = sorted(glob(gt_pth + '/pose/*.txt'), key= lambda x: int(os.path.basename(x)[:-4]))

    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    coord_2d = np.vstack((x_ref, y_ref, np.ones_like(x_ref)))

    with open(os.path.join(gt_pth, '%s.txt' % scene)) as info_f:
        info = [line.rstrip().split(' = ') for line in info_f]
        info = {key: value for key, value in info}
        intrinsics = [
            [float(info['fx_depth']), 0, float(info['mx_depth'])],
            [0, float(info['fy_depth']), float(info['my_depth'])],
            [0, 0, 1]]

    K = torch.tensor(intrinsics).to(device).float()
    src_depth_est = []
    src_extrinsics = []
    geo_pixel_thres = 1.5
    geo_depth_thres = 0.015
    for i, pose_pth in enumerate(pose_list):
        if i % 25 == 0:
            print(total_scenes_index, total_scenes_count, scene, i, len(pose_list))

        frm_name = os.path.basename(pose_pth)[:-4]

        pred_dpth_pth =  os.path.join(dpth_pth, scene, 'refined_depth', frm_name +'.npy')

        if not os.path.isfile(pred_dpth_pth): continue

        pred_dpth =  np.float32(np.load(pred_dpth_pth ).squeeze())
        dpth_prob = np.float32(np.load(pred_dpth_pth.replace('refined_depth', 'refined_prob')).squeeze())
        # pred_dpth[dpth_prob < 0.05] = 0
        pred_depth = cv2.resize(pred_dpth, (width, height), cv2.INTER_LINEAR)
        # pred_dpth = torch.as_tensor(pred_depth).to(device)

        # dpth_prob = np.float32(np.load(pred_dpth_pth).squeeze())
        # pred_depth = cv2.resize(pred_dpth, (width, height), cv2.INTER_LINEAR)

        # dh, dw = pred_dpth.shape

        # K[0] *= dw / width
        # K[1] *= dh / height
        pose = np.loadtxt(pose_pth)
        T = torch.from_numpy(pose).to(device).float()

        if len(src_depth_est) >= 2:
            final_dpth = np.zeros_like(pred_depth)
            val_mask = np.zeros_like(pred_depth)
            for src_dpth , src_T in zip(src_depth_est, src_extrinsics):
                geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(pred_depth, K.cpu().numpy(),
                                                                                            pose,
                                                                                            src_dpth,
                                                                                            K.cpu().numpy(), src_T,
                                                                                            geo_pixel_thres,
                                                                                            geo_depth_thres,
                                                                                            coord_2d)

                final_dpth += depth_reprojected
                val_mask += geo_mask

            final_dpth[val_mask < 2] = 0
            final_dpth[val_mask >=2] /= val_mask[val_mask>=2]
            # final_est = (depth_reprojected + pred_depth) / 2
            # final_est[geo_mask] = 0
            tsdf_fusion.integrate((K @ T.inverse()[:3, :]).to(device),
                                  torch.as_tensor(final_dpth).to(device)
            )
        # else:
        #     final_est = pred_depth

        if len(src_depth_est) < 2:
            src_depth_est.append(pred_depth.copy())
            src_extrinsics.append(pose.copy())
        else:
            src_depth_est = src_depth_est[1:] + [pred_depth.copy()]
            src_extrinsics = src_extrinsics[1:] + [pose.copy()]

    # save trimed mesh
    file_mesh_trim = os.path.join(save_path, '%s_dpth_fuse.ply'%scene)
    tsdf_fusion.get_tsdf().get_mesh('eval')['eval'].export(file_mesh_trim)

    # eval tsdf
    # file_tsdf_trgt = dataset.info['file_name_vol_%02d'%voxel_size]
    # metrics_tsdf = eval_tsdf(file_tsdf_pred, file_tsdf_trgt)

    # eval trimed mesh
    eval_mesh_pth = file_mesh_trim
    file_mesh_trgt = dataset.info['file_name_mesh_gt']
    metrics_mesh, prec_err_pcd, recal_err_pcd = eval_mesh(eval_mesh_pth, file_mesh_trgt) #
    o3d.io.write_point_cloud( os.path.join(save_path,'%s_precErr.ply'%scene), prec_err_pcd)
    o3d.io.write_point_cloud(os.path.join(save_path, '%s_recErr.ply' % scene), recal_err_pcd)

    metrics = { **metrics_mesh}
    print(metrics)

    rslt_file = os.path.join(save_path, '%s_metrics.json'%scene)
    json.dump(metrics, open(rslt_file, 'w'))

    return scene, metrics



def main():
    parser = argparse.ArgumentParser(description="Atlas Testing")
    parser.add_argument("--dataset", default='/data/ScanNet/ScanNet_raw_data/scannet/scans/', metavar="FILE",
                        help="path to checkpoint")
    parser.add_argument("--depth_pred", default='/data/Fengting/ESTDepth_M2/', metavar="FILE",
                        help="path to checkpoint")
    parser.add_argument("--gt_tsdf", default='/data/ScanNet/planeMVS_data/scannet/scans/', metavar="FILE",
                        help="path to checkpoint")
    parser.add_argument("--scenes", default="meta_file/scannet_val.txt",#test
                        help="which scene(s) to run on")
    parser.add_argument("--trim", default=True,#test
                        help="which scene(s) to run on")
    args = parser.parse_args()

    eval_pth = os.path.join(args.depth_pred, '3D_eval')
    if not os.path.isdir(eval_pth):
        os.makedirs(eval_pth)

    # get all the info_file.json's from the command line
    # .txt files contain a list of info_file.json's
    info_files = parse_splits_list(args.scenes)
    # info_files=[info_files[0]]

    metrics = {}
    failed_scene = 0
    for i, info_file in enumerate(info_files):

        # do not if json exists
        scene = os.path.basename(os.path.dirname(info_file))
        rslt_file = os.path.join(args.depth_pred, '3D_eval', '%s_metrics.json' % scene)
        if os.path.isfile(rslt_file):
            temp = json.load(open(rslt_file))
        else:
            # run model on each scene
            gt_pth = os.path.join(args.dataset, scene)
            scene, temp = process(info_file, args.gt_tsdf, gt_pth, args.depth_pred, eval_pth, i, len(info_files))

        # We do not count the scene if it is total failed
        if temp is not None:
            metrics[scene] = temp
        else:
            failed_scene += 1

    rslt_file = os.path.join(args.depth_pred, 'metrics.json')
    json.dump(metrics, open(rslt_file, 'w'))

    # display results
    visualize(rslt_file)
    print('#failed scenes: %d'%failed_scene)
if __name__ == "__main__":
    main()



    # # zip up semseg results for benchmark submission
    # cmd = 'zip -j %s/semseg.zip %s/*.txt'%(save_path, save_path)
    # os.system(cmd)

    # # pretty print metrics
    # print()
    # metrics_keys = list(list(metrics.values())[0].keys())
    # print(''.join( [key.ljust(15) for key in ['scene']+metrics_keys] ))
    # for scene, metrics_i in metrics.items():
    #     metrics_i_fmt = ['%03.3f'%value for value in metrics_i.values()]
    #     print(''.join([s.ljust(15) for s in [scene]+metrics_i_fmt]))
    # metrics_avg = [np.mean([metrics[scene][key] for scene in metrics.keys()]) 
    #                for key in metrics_keys]
    # print()
    # metrics_avg_fmt = ['%03.3f'%value for value in metrics_avg]
    # print(''.join([s.ljust(15) for s in ['average']+metrics_avg_fmt]))
