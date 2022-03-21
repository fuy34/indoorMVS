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
import torch
from torch import nn
from torch.nn import functional as F

from vPlaneRecover.tsdf import coordinates
from vPlaneRecover.backbone3d import conv3dNorm, BasicBlock3d
import numpy as np


class VoxelHeads(nn.Module):
    """ Module that contains all the 3D output heads
    
    Features extracted by the 3D network are passed to this to produce the
    final outputs. Each type of output is added as a head and is responsible
    for returning a dict of outputs and a dict of losses
    """

    def __init__(self, cfg):
        super().__init__()
        self.heads = nn.ModuleList()

        scales = len(cfg.MODEL.BACKBONE3D.CHANNELS) - 1
        final_size = int(cfg.VOXEL_SIZE * 100)
        self.voxel_sizes = [final_size * 2 ** i for i in range(scales)][::-1]


        if "tsdf" in cfg.MODEL.HEADS3D.HEADS:
            self.heads.append(TSDFHead(cfg))

        if "semseg" in cfg.MODEL.HEADS3D.HEADS:
            self.heads.append(SemSegHead(cfg))

        if 'plane_ins' in cfg.MODEL.HEADS3D.HEADS:
            self.heads.append(HTHead(cfg, self.voxel_sizes))


    def forward(self, x, htmap=None, targets=None):
        outputs = {}
        losses = {}

        for head in self.heads:
            out, loss = head(x,  htmap, targets, outputs)
            outputs = { **outputs, **out }
            losses = { **losses, **loss }

        return outputs, losses


class ShrinkageLoss(nn.Module):
    # w.r.t https://openaccess.thecvf.com/content_ECCV_2018/papers/Xiankai_Lu_Deep_Regression_Tracking_ECCV_2018_paper.pdf
    def __init__(self, a=5., c=0.2, reduce=False):
        super(ShrinkageLoss, self).__init__()
        self.a = a
        self.c = c
        self.reduce = reduce

    def forward(self, inputs, targets):
        l = (inputs - targets)
        loss = torch.exp(targets) * (l*l) / \
            (1 + torch.exp(self.a * (self.c - l)))

        if self.reduce:
            return torch.mean(loss)
        else:
            return loss

class FocalLoss(nn.Module):
    # https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289/2
    # focal loss for multi class
    def __init__(self, alpha=3., gamma=2.):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor([alpha, 1-alpha])

    def forward(self, ce_loss, targets=None):
        pt = torch.exp(-ce_loss)
        if targets is None:
            # for multi-class
            at = self.alpha[0].to(ce_loss.device)
            return at * (1 - pt) ** self.gamma * ce_loss
        else:
            # for binary class, we adjust alpha w.r.t. pos/neg sample
            at = self.alpha.to(ce_loss.device).gather(0, targets.data.view(-1)).view(targets.shape)
            return at * (1 - pt) ** self.gamma * ce_loss


class TSDFHead(nn.Module):
    """ Main head that regresses the TSDF"""

    def __init__(self, cfg):
        super().__init__()

        self.loss_weight1 = cfg.MODEL.HEADS3D.TSDF.LOSS_WEIGHT

        self.label_smoothing = cfg.MODEL.HEADS3D.TSDF.LABEL_SMOOTHING

        self.multi_scale = cfg.MODEL.HEADS3D.MULTI_SCALE
        self.split_loss = cfg.MODEL.HEADS3D.TSDF.LOSS_SPLIT #pred
        self.log_transform_loss = cfg.MODEL.HEADS3D.TSDF.LOSS_LOG_TRANSFORM
        self.log_transform_loss_shift = cfg.MODEL.HEADS3D.TSDF.LOSS_LOG_TRANSFORM_SHIFT
        self.sparse_threshold = cfg.MODEL.HEADS3D.TSDF.SPARSE_THRESHOLD

        scales = len(cfg.MODEL.BACKBONE3D.CHANNELS)-1
        final_size = int(cfg.VOXEL_SIZE*100)

        if self.multi_scale:
            self.voxel_sizes = [final_size*2**i for i in range(scales)][::-1]
            classifier = [nn.Conv3d(c, 3, 1, bias=False)
                        for c in cfg.MODEL.BACKBONE3D.CHANNELS[:-1]][::-1] #distinugish the 3 part of a sdf

            decoders = [nn.Conv3d(c+3, 1, 1, bias=False)
                        for c in cfg.MODEL.BACKBONE3D.CHANNELS[:-1]][::-1]
        else:
            self.voxel_sizes = [final_size]
            decoders = [nn.Conv3d(cfg.MODEL.BACKBONE3D.CHANNELS[0], 1, 1, bias=False)]

        self.classifier = nn.ModuleList(classifier)
        self.decoders = nn.ModuleList(decoders)
        self.focal_loss = FocalLoss(alpha=3., gamma=2.)


    def get_normal(self, tsdf_vol):
        # refer to https://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
        # Note the tsdf coordiate are x y z
        # mask = ~torch.logical_or (tsdf_vol == 1, tsdf_vol==-1)
        # replicate usage
        pad_vol = F.pad(tsdf_vol, (1, 1, 1, 1, 1, 1), mode="replicate")  # pad each dim 1,1 to compute grad
        nx = (pad_vol[:,:, 2:, :, :] - pad_vol[:,:, :-2, :, :])[:,:, :, 1:-1, 1:-1]
        ny = (pad_vol[:,:, :, 2:, :] - pad_vol[:,:, :, :-2, :])[:,:, 1:-1, :, 1:-1]
        nz = (pad_vol[:,:, :, :, 2:] - pad_vol[:,:, :, :, :-2])[:,:, 1:-1, 1:-1, :]

        normal = torch.cat([nx, ny, nz], dim=1) # concat in channel dim

        normal /= (normal.norm(dim=1, keepdim=True) + 1e-4) #the change has to be inplace  https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/21
        return normal

    def forward(self, xs,  htmap=None, targets=None, other_outs=None):
        output = {}
        losses = {}
        mask_surface_pred = []
        tsdf_tmp_out = {}

        if not self.multi_scale:
            xs = xs[-1:]


        for i, (clsfer, decoder, x) in enumerate(zip(self.classifier, self.decoders, xs)):
            # pred -1, [-1, 1] and 1 part of tsdf
            parts = clsfer(x)

            # regress the TSDF
            if targets is not None:
                key = 'vol_%02d_tsdf' % self.voxel_sizes[i]
                key3 = 'vol_%02d_parts' % self.voxel_sizes[i]

                trgt = targets[key]
                # get gt parts -1 --> 0(inside), [-1, 1] --> 1 (bdry), 1 --> 2 (outside)
                tgt_part = torch.where(trgt > -1, torch.ones_like(trgt, dtype=torch.long), torch.zeros_like(trgt, dtype=torch.long))
                tgt_part = torch.where(trgt == 1, torch.ones_like(trgt, dtype=torch.long) * 2, tgt_part)
                tsdf_tmp_out[key3] =tgt_part

            parts_in = torch.softmax(parts, dim=1) if not self.training else F.one_hot(tgt_part.squeeze(1), num_classes=3).permute([0,4,1,2,3])
            x_in = torch.cat([parts_in, x], dim=1)
            tsdf = torch.tanh(decoder(x_in)) * self.label_smoothing  # b*1*nx_in*ny_in*nz_in


            tsdf_tmp_out['vol_%02d_tsdf'%self.voxel_sizes[i]] = tsdf #b*1*nx*ny*nz
            output['vol_%02d_parts' % self.voxel_sizes[i]] = parts
            output['vol_%02d_plane_norm' % self.voxel_sizes[i]] = self.get_normal(tsdf.clone())

        # compute losses
        if targets is not None:
            for i, voxel_size in enumerate(self.voxel_sizes):
                key = 'vol_%02d_tsdf'%voxel_size
                pred = tsdf_tmp_out[key]
                trgt = targets[key].to(pred.device)

                key3 = 'vol_%02d_parts' % voxel_size
                tgt_part = tsdf_tmp_out[key3].to(pred.device)
                pred_part = output[key3]

                mask_tgt = tgt_part == 1
                mask_pred = pred_part.argmax(1, keepdim=True) == 1
                # print(tgt_part.shape, mask_tgt.shape, mask_pred.shape, tgt_part.shape, pred_part.shape)
                # exit(1)
                mask = mask_tgt | mask_pred


                part_loss = self.focal_loss(F.cross_entropy(pred_part, tgt_part.squeeze(1), reduction='none')).mean() * 0.5

                if self.log_transform_loss:
                    pred = log_transform(pred, self.log_transform_loss_shift)
                    trgt = log_transform(trgt, self.log_transform_loss_shift)

                tsdf_loss = F.l1_loss(pred, trgt, reduction='none')

                if self.split_loss=='none':
                    losses[key] = tsdf_loss[mask].mean()* self.loss_weight1
                    losses[key3] = part_loss * self.loss_weight1

                elif self.split_loss=='pred':
                    if mask.sum() > 0 and (not part_loss.isnan().any()):

                        losses[key] = tsdf_loss[mask].mean()  * self.loss_weight1
                        losses[key3] = part_loss * self.loss_weight1  if mask.sum() > 0 else 0* part_loss.sum()
                    else:
                        losses[key] = tsdf_loss[mask].sum() * 0.
                        losses[key3] = part_loss.sum() * 0.
                else:
                    raise NotImplementedError("TSDF loss split [%s] not supported"%self.split_loss_empty)

        # update tsdf according to part
        for i, voxel_size in enumerate(self.voxel_sizes):
            output['vol_%02d_tsdf' % voxel_size] =  tsdf_tmp_out['vol_%02d_tsdf' % voxel_size].clone() # prevent modify the pred in place for bp
            pred_inside = output['vol_%02d_parts' % voxel_size].argmax(1, keepdim=True) == 0
            pred_outside = output['vol_%02d_parts' % voxel_size].argmax(1, keepdim=True) == 2
            output['vol_%02d_tsdf' % voxel_size][pred_inside] = -1   # b*1*nx*ny*nz
            output['vol_%02d_tsdf' % voxel_size][pred_outside] = 1

        return output, losses

def log_transform(x, shift=1):
    """ rescales TSDF values to weight voxels near the surface more than close
    to the truncation distance"""
    return x.sign() * (1 + x.abs() / shift).log()


class SemSegHead(nn.Module):
    """ Predicts voxel semantic segmentation"""

    def __init__(self, cfg):
        super().__init__()

        self.multi_scale = cfg.MODEL.HEADS3D.MULTI_SCALE
        self.loss_weight = cfg.MODEL.HEADS3D.SEMSEG.LOSS_WEIGHT

        scales = len(cfg.MODEL.BACKBONE3D.CHANNELS)-1
        final_size = int(cfg.VOXEL_SIZE*100)

        classes = cfg.MODEL.HEADS3D.SEMSEG.NUM_CLASSES
        if self.multi_scale:
            self.voxel_sizes = [final_size*2**i for i in range(scales)][::-1]
            decoders = [nn.Conv3d(c, classes, 1, bias=False) 
                        for c in cfg.MODEL.BACKBONE3D.CHANNELS[:-1]][::-1]
        else:
            self.voxel_sizes = [final_size]
            decoders = [nn.Conv3d(cfg.MODEL.BACKBONE3D.CHANNELS[0], classes, 1, bias=False)]

        self.decoders = nn.ModuleList(decoders)

    def forward(self, xs, htmap = None, targets=None, other_outs=None):
        output = {}
        losses = {}

        if not self.multi_scale:
            xs = xs[-1:] # just use final scale

        for voxel_size, decoder, x in zip(self.voxel_sizes, self.decoders, xs):
            # compute semantic labels
            key = 'vol_%02d_semseg'%voxel_size
            output[key] = decoder(x)  # b*n_class*nx*ny*nz

            # compute losses
            if targets is not None and key in targets:
                pred = output[key]
                trgt = targets[key].to(pred.device)
                mask_surface = targets['vol_%02d_tsdf'%voxel_size].squeeze(1).abs() < 1

                loss = F.cross_entropy(pred, trgt, reduction='none', ignore_index=-1)
                if mask_surface.sum()>0:
                    loss = loss[mask_surface].mean()
                else:
                    loss = 0 * loss.mean()
                losses[key] = loss * self.loss_weight

        return output, losses



class HTHead(nn.Module):
    """ Predicts voxel semantic segmentation"""

    def __init__(self, cfg, voxel_size):
        super().__init__()
        self.voxel_sizes = voxel_size
        self.loss_weight = cfg.MODEL.HEADS3D.HTMAP.LOSS_WEIGHT
        self.shrinkage_loss = ShrinkageLoss(a=2., c=0.2, reduce=False)

    def forward(self, xs, ht_map = None, targets=None, other_outs=None):
        output = {}
        losses = {}


        for voxel_size, pred in zip(self.voxel_sizes, ht_map):
            # compute semantic labels
            if voxel_size == 4 or pred is None: continue

            key = 'vol_%02d_param_htmap'%voxel_size
            mask = torch.ones_like(pred).bool()
            mask[:,:, :, 0, 1:] = False

            # compute losses
            if targets is not None and key in targets:
                # pred = output[key]
                trgt = targets[key].to(pred.device)
                if pred.shape[1] == 1: # no sem dividence case
                    trgt = trgt.sum(1, keepdim=True).clamp(0, 1)
                # loss = F.binary_cross_entropy_with_logits(pred, trgt.type(pred.dtype))
                loss = self.shrinkage_loss(pred, trgt)[mask].mean()

                loss[loss !=loss] = 0
                losses[key] = loss * self.loss_weight

            # remove the theta=0 phi > 0 part
            output[key] = pred.clone()
            output[key][~mask] = 0

        return output, losses




