
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
#  last modification: Fengting Yang 03/21/2022

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


# import thrid_party.Scannet_eval.scannet_eval_util_3d as util_3d

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
        


def process(info_file, save_path, total_scenes_index, total_scenes_count, trim=False):

    # gt depth data loader
    width, height = 640, 480
    transform = transforms.Compose([
        transforms.ResizeImage((width,height)),
        transforms.ToTensor(),
    ])
    dataset = SceneDataset(info_file, transform, frame_types=['depth'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None,
                                             batch_sampler=None, num_workers=2)
    scene = dataset.info['scene']

    # get info about tsdf
    file_tsdf_pred = os.path.join(save_path, '%s.npz'%scene)
    temp = TSDF.load(file_tsdf_pred)
    voxel_size = int(temp.voxel_size*100)
    
    # re-fuse to remove hole filling since filled holes are penalized in 
    # mesh metrics, but do nothing if the hole is not caused by visiablity
    vol_dim = list(temp.tsdf_vol.shape)
    origin = temp.origin
    tsdf_fusion = TSDFFusion(vol_dim, float(voxel_size)/100, origin, color=False)
    device = tsdf_fusion.device

    # mesh renderer
    renderer = Renderer()
    mesh_file = os.path.join(save_path, '%s.ply'%scene) #delte _semseg if test atlas
    mesh = trimesh.load(mesh_file, process=False)
    if isinstance(mesh,trimesh.PointCloud):
        return scene, None

    mesh_opengl = renderer.mesh_opengl(mesh)

    valid_cnt = defaultdict(int)
    metrics_depth =  defaultdict(int)
    for i, d in enumerate(dataloader):
        if i%25==0:
            print(total_scenes_index, total_scenes_count,scene, i, len(dataloader))

        depth_trgt = d['depth'].numpy()
        _, depth_pred = renderer(height, width, d['intrinsics'], d['pose'], mesh_opengl)


        temp = eval_depth(depth_pred, depth_trgt)

        for key, value in temp.items():
            if value != -1:
                metrics_depth[key] += temp[key] # complete will never be -1
                valid_cnt[key] += 1

        # # play video visualizations of depth
        # viz1 = (np.clip((depth_trgt-.5)/5,0,1)*255).astype(np.uint8)
        # viz2 = (np.clip((depth_pred-.5)/5,0,1)*255).astype(np.uint8)
        # viz1 = cv2.applyColorMap(viz1, cv2.COLORMAP_JET)
        # viz2 = cv2.applyColorMap(viz2, cv2.COLORMAP_JET)
        # viz1[depth_trgt==0]=0
        # viz2[depth_pred==0]=0
        # viz = np.hstack((viz1,viz2))
        # cv2.imshow('test', viz)
        # cv2.waitKey(1)

        tsdf_fusion.integrate((d['intrinsics'] @ d['pose'].inverse()[:3,:]).to(device),
                              torch.as_tensor(depth_pred).to(device))


    metrics_depth = {key:value/valid_cnt[key]#len(dataloader)
                     for key, value in metrics_depth.items() if valid_cnt[key] > 0}

    # save trimed mesh
    file_mesh_trim = os.path.join(save_path, '%s_trim.ply'%scene)
    tsdf_fusion.get_tsdf().get_mesh('eval')['eval'].export(file_mesh_trim)

    # eval tsdf
    file_tsdf_trgt = dataset.info['file_name_vol_%02d'%voxel_size]
    metrics_tsdf = eval_tsdf(file_tsdf_pred, file_tsdf_trgt)

    # eval trimed mesh
    eval_mesh_pth = file_mesh_trim if trim else os.path.join(save_path, '%s_plane_ins.ply'%scene)
    file_mesh_trgt = dataset.info['file_name_mesh_gt']
    metrics_mesh, prec_err_pcd, recal_err_pcd = eval_mesh(eval_mesh_pth, file_mesh_trgt) #
    # for debug one
    # o3d.io.write_point_cloud( os.path.join(save_path,'%s_precErr.ply'%scene), prec_err_pcd)
    # o3d.io.write_point_cloud(os.path.join(save_path, '%s_recErr.ply' % scene), recal_err_pcd)

    metrics = {**metrics_depth, **metrics_mesh, **metrics_tsdf}
    print(metrics)

    rslt_file = os.path.join(save_path, '%s_metrics.json'%scene)
    json.dump(metrics, open(rslt_file, 'w'))

    return scene, metrics



def main():
    parser = argparse.ArgumentParser(description="IndoorMVS Eval")
    parser.add_argument("--results", default='', metavar="FILE",
                        help="path to inference results")
    parser.add_argument("--scenes", default="meta_file/scannet_val_demo.txt",
                        help="which scene(s) to run on")
    parser.add_argument("--trim", default=True,
                        help="which scene(s) to run on")
    args = parser.parse_args()

    # get all the info_file.json's from the command line
    # .txt files contain a list of info_file.json's
    info_files = parse_splits_list(args.scenes)

    metrics = {}
    failed_scene = 0
    for i, info_file in enumerate(info_files):

        # do not if json exists
        scene = os.path.basename(os.path.dirname(info_file))
        rslt_file = os.path.join(args.results, '%s_metrics.json' % scene)
        if os.path.isfile(rslt_file):
            temp = json.load(open(rslt_file))
        else:
            # run model on each scene
            scene, temp = process(info_file, args.results, i, len(info_files), trim=args.trim)

        # We do not count the scene if it is total failed
        if temp is not None:
            metrics[scene] = temp
        else:
            failed_scene += 1

    rslt_file = os.path.join(args.results, 'metrics.json')
    json.dump(metrics, open(rslt_file, 'w'))

    # display results
    visualize(rslt_file)
    print('#failed scenes: %d'%failed_scene)

if __name__ == "__main__":
    main()
