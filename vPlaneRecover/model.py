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

import itertools
import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from vPlaneRecover.config import CfgNode
from vPlaneRecover.data import ScenesDataset, collate_fn, parse_splits_list
from vPlaneRecover.heads3d_heatmapCls_now import VoxelHeads #heatmapCls_now
from vPlaneRecover.featExactor import build_backbone2d
from vPlaneRecover.backbone3d import build_backbone3d
import vPlaneRecover.transforms as transforms
from vPlaneRecover.tsdf import coordinates, TSDF
from vPlaneRecover.util import get_planeIns_htmap, get_planeInsVert_frmHT
from glob import glob

from skimage import measure
import matplotlib.colors as plcolors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import trimesh
import numpy as np

def backproject(voxel_dim, voxel_size, origin, projection, features):
    """ Takes 2d features and fills them along rays in a 3d volume

    This function implements eqs. 1,2 in https://arxiv.org/pdf/2003.10432.pdf
    Each pixel in a feature image corresponds to a ray in 3d.
    We fill all the voxels along the ray with that pixel's features.

    Args:
        voxel_dim: size of voxel volume to construct (nx,ny,nz)
        voxel_size: metric size of each voxel (ex: .04m)
        origin: origin of the voxel volume (xyz position of voxel (0,0,0))
        projection: bx4x3 projection matrices (intrinsics@extrinsics)
        features: bxcxhxw  2d feature tensor to be backprojected into 3d

    Returns:
        volume: b x c x nx x ny x nz 3d feature volume
        valid:  b x 1 x nx x ny x nz volume.
                Each voxel contains a 1 if it projects to a pixel
                and 0 otherwise (not in view frustrum of the camera)
    """
    batch = features.size(0)
    channels = features.size(1)
    device = features.device
    nx, ny, nz = voxel_dim

    coords = coordinates(voxel_dim, device).unsqueeze(0).expand(batch,-1,-1) # b*3* (nx*ny*nz)
    world = coords.type_as(projection) * voxel_size + origin.to(device).unsqueeze(2) #todo: adjust proj scale
    world = torch.cat((world, torch.ones_like(world[:,:1]) ), dim=1)
    
    camera = torch.bmm(projection, world) # world coord val in camera coord system
    px = (camera[:,0,:]/camera[:,2,:]).round().type(torch.long)
    py = (camera[:,1,:]/camera[:,2,:]).round().type(torch.long)
    pz = camera[:,2,:]

    # voxels in view frustrum
    height, width = features.size()[2:]
    valid = (px >= 0) & (py >= 0) & (px < width) & (py < height) & (pz>0) # bxhwd

    # put features in volume
    volume = torch.zeros(batch, channels, nx*ny*nz, dtype=features.dtype,  device=device)
    for b in range(batch):
        volume[b,:,valid[b]] = features[b,:,py[b,valid[b]], px[b,valid[b]]]

    volume = volume.view(batch, channels, nx, ny, nz)
    valid = valid.view(batch, 1, nx, ny, nz) # recorded the whole frusture area

    return volume, valid


class vPlaneRecNet(pl.LightningModule):
    """ Network architecture implementing ATLAS (https://arxiv.org/pdf/2003.10432.pdf)"""

    def __init__(self, hparams):
        super().__init__()

        # see config.py for details .convert_to_dict()
        # self.hparams = hparams
        self.hparams.update(hparams)
        self.save_hyperparameters()

        # pytorch lightning does not support saving YACS CfgNone     
        self.cfg = CfgNode(self.hparams)
        cfg = self.cfg

        # networks
        self.backbone2d,  self.backbone2d_stride = build_backbone2d(cfg)
        self.backbone3d = build_backbone3d(cfg)
        # self.heads2d = PixelHeads(cfg, self.backbone2d_stride) #ModuleList(), empty did not use actually
        self.heads3d = VoxelHeads(cfg)

        # other hparams
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)

        self.voxel_size = cfg.VOXEL_SIZE
        self.voxel_dim_train = cfg.VOXEL_DIM_TRAIN
        self.voxel_dim_val = cfg.VOXEL_DIM_VAL
        # self.voxel_dim_test = cfg.VOXEL_DIM_TEST
        self.origin = torch.tensor([0,0,0]).view(1,3)

        self.batch_size_train = cfg.DATA.BATCH_SIZE_TRAIN
        self.num_frames_train = cfg.DATA.NUM_FRAMES_TRAIN
        self.num_frames_val = cfg.DATA.NUM_FRAMES_VAL
        self.frame_types = cfg.MODEL.HEADS2D.HEADS
        self.frame_selection = cfg.DATA.FRAME_SELECTION
        self.batch_backbone2d_time = cfg.TRAINER.BATCH_BACKBONE2D_TIME
        self.finetune3d = cfg.TRAINER.FINETUNE3D
        self.voxel_types = cfg.MODEL.HEADS3D.HEADS
        self.voxel_sizes = [int(cfg.VOXEL_SIZE*100)*2**i for i in 
                            range(len(cfg.MODEL.BACKBONE3D.LAYERS_DOWN)-1)] #_DOWN

        self.log_pth =os.path.join(cfg.LOG_DIR, cfg.TRAINER.NAME, cfg.TRAINER.VERSION +
                                   '_lr{}_bz{}_ep{}_nfrm{}_resnet{}'.format(cfg.OPTIMIZER.ADAM.LR,
                                         int( cfg.DATA.BATCH_SIZE_TRAIN * cfg.TRAINER.NUM_GPUS),
                                                                            cfg.TRAINER.NUM_EPOCH,
                                                                            cfg.DATA.NUM_FRAMES_TRAIN,
                                                                            cfg.MODEL.RESNETS.DEPTH))

        self.initialize_volume()


    def initialize_volume(self):
        """ Reset the accumulators.
        
        self.volume is a voxel volume containg the accumulated features
        self.valid is a voxel volume containg the number of times a voxel has
            been seen by a camera view frustrum
        """

        self.volume = 0
        self.valid = 0

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def inference1(self, projection_in, image=None, feature=None):
        """ Backprojects image features into 3D and accumulates them.

        This is the first half of the network which is run on every frame.
        Only pass one of image or feature. If image is passed 2D features
        are extracted from the image using self.backbone2d. When features
        are extracted external to this function pass features (used when 
        passing multiple frames through the backbone2d simultaniously
        to share BatchNorm stats).

        Args:
            projection: bx3x4 projection matrix
            image: bx3xhxw RGB image
            feature: bxcxh'xw' feature map (h'=h/stride, w'=w/stride)

        Feature volume is accumulated into self.volume and self.valid
        """

        assert ((image is not None and feature is None) or 
                (image is None and feature is not None))

        if feature is None:
            image = self.normalizer(image)
            feature = self.backbone2d(image) # b * 32 * 120 * 160
            # feature_lst = feature_lst[::-1]  # change to p5 p4 p3 p2

        # backbone2d reduces the size of the images so we
        # change intrinsics to reflect this
        projection = projection_in.clone()
        projection[:, :2, :] = projection[:, :2, :] / self.backbone2d_stride

        if self.training:
            voxel_dim = self.voxel_dim_train
        else:
            voxel_dim = self.voxel_dim_val
        volume, valid = backproject(voxel_dim, self.voxel_size, self.origin,
                                    projection, feature)

        if self.finetune3d:
            volume.detach_()
            valid.detach_()

        self.volume = self.volume + volume
        self.valid = self.valid + valid


    def inference2(self, targets=None):
        """ Refines accumulated features and regresses output TSDF.

        This is the second half of the network. It should be run once after
        all frames have been accumulated. It may also be run more fequently
        to visualize incremental progress.

        Args:
            targets: used to compare network output to ground truth

        Returns:
            tuple of dicts ({outputs}, {losses})
                if targets is None, losses is empty
        """
        volume = self.volume / self.valid #normalize

        # remove nans (where self.valid==0)
        volume = volume.transpose(0, 1)
        volume[:, self.valid.squeeze(1) == 0] = 0
        volume = volume.transpose(0, 1)

        x, htmaps = self.backbone3d(volume)  # a list feature, 0: b*128*(nx/4)*(ny/4)*(nz/4), 1: 64*(nk/2),  2: b*32*nx*ny*nz

        return self.heads3d(x, htmaps, targets)


    def forward(self, batch):
        """ Wraps inference1() and inference2() into a single call.

        Args:
            batch: a dict from the dataloader

        Returns:
            see self.inference2
        """

        self.initialize_volume()

        image = batch['image'] # n_img * 1 * 3 * h* w, why 1 ?
        projection = batch['projection']

        # get targets if they are in the batch
        targets3d = {key:value for key, value in batch.items() if key[:3]=='vol'}
        targets3d = targets3d if targets3d else None


        # transpose batch and time so we can accumulate sequentially
        images = image.transpose(0,1)
        projections = projection.transpose(0,1)

        if (not self.batch_backbone2d_time) or (not self.training) or self.finetune3d:
            # run images through 2d cnn sequentially and backproject and accumulate
            for image, projection in zip(images, projections):
                self.inference1(projection, image=image)

        else:
            # run all images through 2d cnn together to share batchnorm stats
            image = images.reshape(images.shape[0]*images.shape[1], *images.shape[2:])
            image = self.normalizer(image)
            features = self.backbone2d(image)  # c_feat * 0.25H*0.25W

            features = features.view(images.shape[0],
                                     images.shape[1],
                                     *features.shape[1:])  # n_img * 1 * 3 * h* w

            for projection, feature in zip(projections, features):
                self.inference1(projection, feature=feature)

        # run 3d cnn
        outputs3d, losses3d = self.inference2(targets3d)

        return {**outputs3d}, { **losses3d} #**outputs2d, **losses2d,


    def postprocess(self, batch, name='pred', b_val=False):
        """ Wraps the network output into a TSDF data structure
        
        Args:
            batch: dict containg network outputs

        Returns:
            list of TSDFs (one TSDF per scene in the batch)
        """
        
        key = 'vol_%02d'%self.voxel_sizes[0] # only get vol of final resolution
        out = []
        batch_size = len(batch[key+'_tsdf'])

        for i in range(batch_size):
            tsdf = TSDF(self.voxel_size, 
                        self.origin,
                        batch[key+'_tsdf'][i].squeeze(0)) # build a obj contain the final tsdf volume

            if key + '_plane_norm' in batch:
                tsdf.attribute_vols['plane_norm'] = batch[key + '_plane_norm'][i]

            if key + '_plane_cls' in batch:
                tsdf.attribute_vols['plane_cls'] = batch[key + '_plane_cls'][i]

            if key + '_plane_ins' in batch:
                tsdf.attribute_vols['plane_ins'] = batch[key + '_plane_ins'][i]

            if 'vol_%02d' % self.voxel_sizes[1] + '_param_htmap' in batch:
                tsdf.attribute_vols['param_htmap'] = batch['vol_%02d' % self.voxel_sizes[1] + '_param_htmap'][i]


            # add semseg vol
            if ('semseg' in self.voxel_types) and (key+'_semseg' in batch):
                semseg = batch[key+'_semseg'][i]
                if semseg.ndim == 4:
                    semseg = semseg.argmax(0)  # nx * ny * nz, with one of label index
                tsdf.attribute_vols['semseg'] = semseg


            # add color vol
            if 'color' in self.voxel_types:
                color = batch[key+'_color'][i]
                tsdf.attribute_vols['color'] = color

            # more attri.
            if 'cenprob' in self.voxel_types:
                tsdf.attribute_vols['centroid_prob'] = batch[key + '_centroid_prob'][i].squeeze(0)

            # cluster plane instance
            if ('semseg' in self.voxel_types) and ('plane_ins' in self.voxel_types):
                key2 = 'vol_%02d' % self.voxel_sizes[1]

                param_htmap = batch[key2 + '_param_htmap'][i]

                if ('trgt' not in name):
                    if b_val :
                        tsdf.attribute_vols['plane_ins_vert'] = get_planeInsVert_frmHT(batch[key + '_tsdf'][i],self.voxel_size,  self.origin,
                                                          semseg, param_htmap,self.backbone3d.vote_idxs[-1],
                                                          self.backbone3d.rhos[-1], self.backbone3d.thetas[-1], self.backbone3d.phis[-1], self.cfg)
                else:
                    tsdf.attribute_vols['plane_ins'] = batch[key + '_plane_ins'][i]


            out.append(tsdf)

        return out


    def get_transform(self, is_train):
        """ Gets a transform to preprocess the input data"""

        if is_train:
            voxel_dim = self.voxel_dim_train
            random_rotation = self.cfg.DATA.RANDOM_ROTATION_3D
            random_translation = self.cfg.DATA.RANDOM_TRANSLATION_3D
            paddingXY = self.cfg.DATA.PAD_XY_3D
            paddingZ = self.cfg.DATA.PAD_Z_3D
        else:
            # center volume
            voxel_dim = self.voxel_dim_val
            random_rotation = False
            random_translation = False
            paddingXY = 0
            paddingZ = 0

        transform = []
        transform += [transforms.ResizeImage((640,480)),    # read frames and resize them + intrinsic
                      transforms.ToTensor(),                # put image, intrinsic , pose into torch tensor
                      transforms.InstanceToSemseg('nyu40'), # map 3D instance to nyu label,
                      transforms.RandomTransformSpace(
                          voxel_dim, random_rotation, random_translation,
                          paddingXY, paddingZ), # this is the key transform adjust the tsdf size (cropping and interpolation) !
                      transforms.FlattenTSDF(),
                      transforms.IntrinsicsPoseToProjection(),
                     ]

        return transforms.Compose(transform)


    def train_dataloader(self):
        transform = self.get_transform(True)
        info_files = parse_splits_list(self.cfg.DATASETS_TRAIN)
        dataset = ScenesDataset(
            info_files, self.num_frames_train, transform,
            self.frame_types, self.frame_selection, self.voxel_types,
            self.voxel_sizes)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size_train, num_workers=self.cfg.TRAINER.NUM_DATALOADER,
            collate_fn=collate_fn, shuffle=True, drop_last=True)
        return dataloader

    def val_dataloader(self):
        transform = self.get_transform(False)
        info_files = parse_splits_list(self.cfg.DATASETS_VAL)
        dataset = ScenesDataset(
            info_files, self.num_frames_val, transform,
            self.frame_types, self.frame_selection, self.voxel_types,
            self.voxel_sizes)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=1, collate_fn=collate_fn,
            shuffle=False, drop_last=False)
        return dataloader


    def training_step(self, batch, batch_idx):

        outputs, losses = self.forward(batch)

        # for debug only -- if there is no tsdf branch
        key = 'vol_%02d' % self.voxel_sizes[0]

        # visualize training outputs at the begining of each epoch, after the first checkpoint save period
        if batch_idx==0 : #and self.current_epoch > self.cfg.TRAINER.CHECKPOINT_PERIOD:

            pred_tsdfs = self.postprocess(outputs, 'pred') #batch['scene'][0]+'_pred_ep{}'.format(self.current_epoch)
            trgt_tsdfs = self.postprocess(batch,  'trgt') #batch['scene'][0]+'_trgt_ep{}'.format(self.current_epoch)
            self.logger.experiment1.save_mesh(pred_tsdfs[0], 'train_pred')
            self.logger.experiment1.save_mesh(trgt_tsdfs[0], 'train_trgt')

            # for debug only
            # if (key+'_parts') in outputs:
            #     tsdf = TSDF(self.voxel_size,
            #                     self.origin,
            #                     batch[key + '_tsdf'][0].squeeze(0))
            #
            #     tsdf.attribute_vols['parts'] = outputs[key+'_parts'][0].argmax(0) #tgt_part#
            #     part_viz = tsdf.get_mesh('parts')
            #     # save_path = os.path.join(self.cfg.LOG_DIR, self.cfg.TRAINER.NAME, self.cfg.TRAINER.VERSION)
            #     if isinstance(part_viz, dict):
            #         part_viz['parts'].export(self.log_pth + '/parts.ply')
            #     else:  # in case the whole gt is out of view
            #         part_viz.export(self.log_pth + '/empty.ply')

        self.log_dict(losses)
        loss = sum(losses.values())

        # print log -- for debug only
        # tsdf_loss = sum([losses[x] for x in losses.keys() if 'tsdf' in x])
        # part_loss = sum([losses[x] for x in losses.keys() if 'parts' in x])
        # semg_loss = sum([losses[x] for x in losses.keys() if 'semseg' in x])
        # htmap_loss = sum([losses[x] for x in losses.keys() if '_param_htmap' in x])
        # print('tsdf {:.3f}, tsdf_cls {:.3f}, semg {:.3f}, htmap {:.3f}, lr {}'.format(
        #         tsdf_loss, part_loss, semg_loss,  htmap_loss, self.trainer.lr_schedulers[0]['scheduler']._last_lr))

        return {'loss': loss, 'log': losses}


    def validation_step(self, batch, batch_idx):
        outputs, losses = self.forward(batch)


        key = 'vol_%02d' % self.voxel_sizes[0]
        if (key + '_tsdf') not in outputs:
            outputs[key + '_tsdf'] = batch[key + '_tsdf']


        # save validation meshes, only viz the first 5 to save memory
        if batch_idx < 1:
            pred_tsdfs = self.postprocess(outputs, 'pred', b_val=True)
            trgt_tsdfs = self.postprocess(batch,'trgt')
            self.logger.experiment1.save_mesh(pred_tsdfs[0],
                                              batch['scene'][0]+'_pred_ep{}'.format(self.current_epoch))
            self.logger.experiment1.save_mesh(trgt_tsdfs[0],
                                              batch['scene'][0] + '_trgt'.format(self.current_epoch))

        return losses

    def validation_epoch_end(self, outputs):
        avg_losses = {'val_'+key:torch.stack([x[key] for x in outputs]).mean() 
                      for key in outputs[0].keys()}
        avg_loss = sum(avg_losses.values())

        # record ckpt
        ckpt_lst = glob(self.log_pth + '/epoch*.ckpt')
        ckpt_lst.sort()
        remain_last = 10
        if len(ckpt_lst) > remain_last:
            os.remove(ckpt_lst[0])
            print('len', len(ckpt_lst), 'remove', ckpt_lst[0])

        self.log_dict(avg_losses)
        return {'val_loss': avg_loss, 'log': avg_losses}


    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        # allow for different learning rates between pretrained layers 
        # (resnet backbone) and new layers (everything else).
        # params_backbone2d = self.backbone2d[0].parameters()
        params_backbone2d =self.backbone2d.parameters() #[x for x in self.backbone2d.parameters() if x not in self.backbone2d.freezon_param]
        modules_rest = [ self.backbone3d,
                        self.heads3d] # self.heads2d,
        # itertools.chain: Make an iterator that returns elements from the first iterable until it is exhausted,
        # then proceeds to the next iterable, until all of the iterables are exhausted
        params_rest = itertools.chain(*(params.parameters() 
                                        for params in modules_rest))

        # optimzer
        if self.cfg.OPTIMIZER.TYPE == 'Adam':
            lr = self.cfg.OPTIMIZER.ADAM.LR
            lr_backbone2d = lr * self.cfg.OPTIMIZER.BACKBONE2D_LR_FACTOR #factor = 1
            optimizer = torch.optim.Adam([
                {'params': params_backbone2d, 'lr': lr_backbone2d},
                {'params': params_rest, 'lr': lr}]) # , amsgrad=True
            optimizers.append(optimizer)

        else:
            raise NotImplementedError(
                'optimizer %s not supported'%self.cfg.OPTIMIZER.TYPE)

        # scheduler, seems try to divide lr between backbone and later, but give up
        if self.cfg.SCHEDULER.TYPE == 'StepLR':
            # Decays the learning rate of each parameter group by gamma every step_size epochs
            # here lr decayed after 300 epoch, almost never happened
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, self.cfg.SCHEDULER.STEP_LR.STEP_SIZE,
                gamma=self.cfg.SCHEDULER.STEP_LR.GAMMA)
            schedulers.append(scheduler)

        elif self.cfg.SCHEDULER.TYPE != 'None':
            raise NotImplementedError(
                'optimizer %s not supported'%self.cfg.OPTIMIZER.TYPE)
                
        return optimizers, schedulers




    

