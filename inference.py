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
import os

import numpy as np
import torch

from vPlaneRecover.data import SceneDataset, parse_splits_list
from vPlaneRecover.model import vPlaneRecNet
import vPlaneRecover.transforms as transforms
from vPlaneRecover.evaluation import project_to_mesh
import third_party.Scannet_eval.scannet_eval_util_3d as util_3d

import trimesh
from vPlaneRecover.backbone3d import build_backbone3d

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def process(info_file, model, num_frames, save_path, total_scenes_index, total_scenes_count):
    """ Run the netork on a scene and save output

    Args:
        info_file: path to info_json file for the scene
        model: pytorch model that implemets Atlas
        frames: number of frames to use in reconstruction (-1 for all)
        save_path: where to save outputs
        total_scenes_index: used to print which scene we are on
        total_scenes_count: used to print the total number of scenes to process
    """
    # do not inference twice if already there
    cur_scene = os.path.basename(os.path.dirname(info_file))
    if cur_scene[-2:] != '00':return

    if os.path.isfile(os.path.join(save_path, '%s.npz'%cur_scene)):
        return

    voxel_scale = model.voxel_sizes[0]
    dataset = SceneDataset(info_file, voxel_sizes=[voxel_scale],
                           voxel_types=model.voxel_types, num_frames=num_frames)

    # compute voxel origin
    if 'file_name_vol_%02d'%voxel_scale in dataset.info:
        # compute voxel origin from ground truth
        tsdf_trgt = dataset.get_tsdf()['vol_%02d'%voxel_scale]
        voxel_size = float(voxel_scale)/100
        # shift by integer number of voxels for padding
        shift = torch.tensor([.5, .5, .5])//voxel_size
        offset = tsdf_trgt.origin - shift*voxel_size

    else:
        # use default origin
        # assume floor is a z=0 so pad bottom a bit
        offset = torch.tensor([0,0,-.5])
    T = torch.eye(4)
    T[:3,3] = offset

    # insert transformation after dataset init
    transform = transforms.Compose([
        transforms.ResizeImage((640,480)),
        transforms.ToTensor(),
        transforms.TransformSpace(T, model.voxel_dim_val, [0,0,0]),
        transforms.IntrinsicsPoseToProjection(),
    ])
    dataset.transform = transform
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None,
                                             batch_sampler=None, num_workers=2)

    scene = dataset.info['scene']

    model.initialize_volume()
    torch.cuda.empty_cache()

    for j, d in enumerate(dataloader):
        # print(d.keys()) #file_name_image, image, projection
        # logging progress
        if j%25==0:
            print(total_scenes_index,
                  total_scenes_count,
                  dataset.info['dataset'],
                  scene,
                  j,
                  len(dataloader)
            )
        model.inference1(d['projection'].unsqueeze(0).cuda(),
                         image=d['image'].unsqueeze(0).cuda())
    torch.cuda.empty_cache()
    outputs, losses = model.inference2()

    # provide gt as tsdf result for debug
    if 'vol_%02d_tsdf'%voxel_scale not in outputs:
        T = torch.eye(4)
        T[:3, 3] = offset
        transform = transforms.Compose([
            transforms.ResizeImage((640, 480)),
            transforms.ToTensor(),
            transforms.TransformSpace(T, model.voxel_dim_val, [0, 0, 0]),
            transforms.IntrinsicsPoseToProjection(),
        ])

        dataset.transform = transform
        tsdf_trgt = dataset.get_tsdf()['vol_%02d' % voxel_scale]
        tsdf_vol = tsdf_trgt.tsdf_vol.detach().clone()

        outputs['vol_%02d_tsdf'%voxel_scale] = tsdf_vol.unsqueeze(0).unsqueeze(0)

    tsdf_pred = model.postprocess(outputs, b_val=True)[0]

    # TODO: set origin in model... make consistent with offset above?
    tsdf_pred.origin = offset.view(1,3).cuda()
   
    output_meshs = []
    if 'semseg' in tsdf_pred.attribute_vols:
        output_meshs.append('semseg')
        output_meshs.append('semseg_ent')

    if 'centroid_prob' in tsdf_pred.attribute_vols:
        output_meshs.append('centroid_prob')

    if 'plane_ins' in tsdf_pred.attribute_vols:
        output_meshs.append('plane_ins')
        output_meshs.append('vert_plane')
        output_meshs.append('plane_cls')

    meshes = tsdf_pred.get_mesh(output_meshs)
    attribute_mesh = None
    if isinstance(meshes, dict):
        for key in meshes:
            if key == 'semseg':
                meshes[key].export(os.path.join(save_path, '%s.ply' % scene)) #_semseg
                # save vertex attributes seperately since trimesh doesn't
                np.savez(os.path.join(save_path, '%s_attributes.npz'%scene),
                        **(meshes[key]).vertex_attributes)
                attribute_mesh = meshes[key]

            else:
                meshes[key].export(os.path.join(save_path, '%s_%s.ply' %(scene, key)))
    else:
        meshes.export(os.path.join(save_path, '%s.ply' %(scene)))
    tsdf_pred.save(os.path.join(save_path, '%s.npz'%scene))

    # transfer semantic txt and instance txt for evaluation
    file_mesh_trgt = dataset.info['file_name_mesh_gt']
    if attribute_mesh is not None:
        # save as txt for benchmark evaluation
        mesh_trgt = trimesh.load(file_mesh_trgt, process=False)
        mesh_transfer = project_to_mesh(attribute_mesh, mesh_trgt, 'semseg')
        semseg = mesh_transfer.vertex_attributes['semseg']
        sem_save_pth = os.path.join(save_path, 'semseg')
        if not os.path.isdir(sem_save_pth):
            os.makedirs(sem_save_pth)
        np.savetxt(os.path.join(sem_save_pth, '%s.txt' % scene), semseg, fmt='%d')
        mesh_transfer.export(os.path.join(sem_save_pth, '%s_transfer.ply' % scene))

        # save plane instance label-- note the plane_ins attribute is only stored in mesh, we use mesh_planeIns to offer color
        if os.path.isfile(os.path.join(save_path, '%s_plane_ins.ply' % scene)):
            mesh_planeIns_gt = trimesh.load(dataset.info['file_name_plane_mesh'], process=False)
            mesh_planeIns_pred = trimesh.load(os.path.join(save_path, '%s_plane_ins.ply' % scene), process=False)
            mesh_planeIns_transfer = project_to_mesh(attribute_mesh, mesh_planeIns_gt, 'plane_ins', mesh_planeIns_pred)
            planeIns = mesh_planeIns_transfer.vertex_attributes['plane_ins']
            plnIns_save_pth = os.path.join(save_path, 'plane_ins')
            if not os.path.isdir(plnIns_save_pth):
                os.makedirs(plnIns_save_pth)

            mesh_planeIns_transfer.export(os.path.join(plnIns_save_pth, '%s_planeIns_transfer.ply' % scene))
            util_3d.export_instance_ids_for_eval(os.path.join(plnIns_save_pth, '%s.txt' % scene), (semseg), planeIns)



def main():
    parser = argparse.ArgumentParser(description="IndoorMVS Inference")
    parser.add_argument("--model", default='/data/Fengting/vPlaneRecover_train/vPlaneRecover/HT_sepPartNormHT_newthre06_lr0.0005_bz4_ep150_nfrm50_resnet50/epoch=134_step=00030104.ckpt', metavar="FILE",
                        help="path to checkpoint")
    parser.add_argument("--scenes", default='meta_file/scannet_val_demo.txt',
                        help="which scene(s) to run on")
    parser.add_argument("--num_frames", default=-1, type=int,
                        help="number of frames to use (-1 for all)")
    parser.add_argument("--save_path", default='val',  help="path to save result")
    parser.add_argument("--topk", default=int(8e6), type=int, help="number of topk center prob to be used -- ignore")
    parser.add_argument("--heatmap_thres", default=0.008, type=float,   help="Threshold for heatmap plane detection")
    parser.add_argument("--voxel_dim", nargs=3, default=[256,256,128], type=int,  help="override voxel dim")
    args = parser.parse_args()

    # get all the info_file.json's from the command line
    # .txt files contain a list of info_file.json's
    info_files = parse_splits_list(args.scenes)

    model = vPlaneRecNet.load_from_checkpoint(args.model) # all hyper-param setting is in torch.load(args.model)['hyper_parameters']
    model = model.cuda().eval()
    torch.set_grad_enabled(False)

    # overwrite default values of voxel_dim_test
    if args.voxel_dim[0] != -1:
        model.voxel_dim_test = args.voxel_dim
        model.cfg.VOXEL_DIM_VAL = args.voxel_dim
        model.backbone3d.voxel_dim_val =  args.voxel_dim

    # TODO: implement voxel_dim_test
    model.voxel_dim_val = model.voxel_dim_test
    model.cfg.MODEL.GROUPING.TOPK_PROB = args.topk # useless
    model.cfg.MODEL.GROUPING.PROB_THRES = args.heatmap_thres
    model_name = os.path.splitext(os.path.split(args.model)[1])[0]
    if 'test' in args.scenes : # not used in our work
        model.voxel_types = ['tsdf', 'semseg']
        save_path = os.path.join(model.cfg.LOG_DIR, model.cfg.TRAINER.NAME,
                                 model.cfg.TRAINER.VERSION, 'test_{}_'.format(args.heatmap_thres) + model_name)
    else:
        save_path = os.path.join(model.cfg.LOG_DIR, model.cfg.TRAINER.NAME,
                                 model.cfg.TRAINER.VERSION, 'val_{}_'.format(args.heatmap_thres) + model_name) #args.save_path

    if args.num_frames>-1:
        save_path = '%s_%d'%(save_path, args.num_frames)
    os.makedirs(save_path, exist_ok=True)

    for i, info_file in enumerate(info_files):
        # run model on each scene
        with torch.no_grad():
            process(info_file, model, args.num_frames, save_path, i, len(info_files))

if __name__ == "__main__":
    main()
