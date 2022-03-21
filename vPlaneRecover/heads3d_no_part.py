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

import torch
from torch import nn
from torch.nn import functional as F

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

        if "tsdf" in cfg.MODEL.HEADS3D.HEADS:
            self.heads.append(TSDFHead(cfg))

        if "semseg" in cfg.MODEL.HEADS3D.HEADS:
            self.heads.append(SemSegHead(cfg))

        # if "cenprob" in cfg.MODEL.HEADS3D.HEADS:
        #     self.heads.append(CenProbHead(cfg))

        # if "color" in cfg.MODEL.HEADS3D.HEADS:
        #     self.heads.append(ColorHead(cfg))


    def forward(self, x, htmap=None, targets=None):
        outputs = {}
        losses = {}

        for head in self.heads:
            out, loss = head(x, targets)
            outputs = { **outputs, **out }
            losses = { **losses, **loss }

        return outputs, losses


class TSDFHead(nn.Module):
    """ Main head that regresses the TSDF"""

    def __init__(self, cfg):
        super().__init__()

        self.loss_weight = cfg.MODEL.HEADS3D.TSDF.LOSS_WEIGHT
        # self.loss_weight2 = cfg.MODEL.HEADS3D.NORM.LOSS_WEIGHT

        self.label_smoothing = cfg.MODEL.HEADS3D.TSDF.LABEL_SMOOTHING

        self.multi_scale = cfg.MODEL.HEADS3D.MULTI_SCALE
        self.split_loss = cfg.MODEL.HEADS3D.TSDF.LOSS_SPLIT
        self.log_transform_loss = cfg.MODEL.HEADS3D.TSDF.LOSS_LOG_TRANSFORM
        self.log_transform_loss_shift = cfg.MODEL.HEADS3D.TSDF.LOSS_LOG_TRANSFORM_SHIFT
        self.sparse_threshold = cfg.MODEL.HEADS3D.TSDF.SPARSE_THRESHOLD

        scales = len(cfg.MODEL.BACKBONE3D.CHANNELS)-1
        final_size = int(cfg.VOXEL_SIZE*100)

        if self.multi_scale:
            self.voxel_sizes = [final_size*2**i for i in range(scales)][::-1]
            # classifier = [nn.Conv3d(c, 3, 1, bias=False)
            #             for c in cfg.MODEL.BACKBONE3D.CHANNELS[:-1]][::-1] #distinugish the 3 part of a sdf

            decoders = [nn.Conv3d(c, 1, 1, bias=False)
                        for c in cfg.MODEL.BACKBONE3D.CHANNELS[:-1]][::-1]
        else:
            self.voxel_sizes = [final_size]
            decoders = [nn.Conv3d(cfg.MODEL.BACKBONE3D.CHANNELS[0], 1, 1, bias=False)]

        # self.classifier = nn.ModuleList(classifier)
        self.decoders = nn.ModuleList(decoders)

    def focal_loss(self,ce_loss, alpha=3., gamma=2.):
        # https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289/2
        # focal loss for multi class
        pt = torch.exp(-ce_loss)
        return (alpha * (1 - pt) ** gamma * ce_loss)

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

        normal /= (normal.norm(dim=1) + 1e-4) #the change has to be inplace  https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/21
        # we cannot use the one below, it will lead to non-differialable
        # normal[normal != normal] = 0  # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4 set nan to 0
        return normal

    def forward(self, xs, targets=None):
        output = {}
        losses = {}
        mask_surface_pred = []

        if not self.multi_scale:
            xs = xs[-1:]


        for i, ( decoder, x) in enumerate(zip( self.decoders, xs)):
            # regress the TSDF
            tsdf = torch.tanh(decoder(x)) * self.label_smoothing  # b*1*nx_in*ny_in*nz_in

            # use previous scale to sparsify current scale
            if self.split_loss == 'pred' and i > 0:
                tsdf_prev = output['vol_%02d_tsdf' % self.voxel_sizes[i - 1]]
                tsdf_prev = F.interpolate(tsdf_prev, scale_factor=2)
                # FIXME: when using float16, why is interpolate casting to float32?
                tsdf_prev = tsdf_prev.type_as(tsdf)
                mask_surface_pred_prev = tsdf_prev.abs() < self.sparse_threshold[i - 1]
                # .999 so we don't close surfaces during mc
                # if the voxel is invalid in the previous output, it will be invalid in current as well
                tsdf[~mask_surface_pred_prev] = tsdf_prev[~mask_surface_pred_prev].sign() * .999
                mask_surface_pred.append(mask_surface_pred_prev)

            output['vol_%02d_tsdf' % self.voxel_sizes[i]] = tsdf  # b*1*nx*ny*nz

            # compute losses
        if targets is not None:
            for i, voxel_size in enumerate(self.voxel_sizes):
                key = 'vol_%02d_tsdf' % voxel_size
                pred = output[key]
                trgt = targets[key]

                mask_observed = trgt < 1
                mask_outside = (trgt == 1).all(-1, keepdim=True)

                # TODO: extend mask_outside (in heads:loss) to also include
                # below floor... maybe modify padding_mode in tsdf.transform...
                # probably cleaner to look along slices similar to how we look
                # along columns for outside.

                if self.log_transform_loss:
                    pred = log_transform(pred, self.log_transform_loss_shift)
                    trgt = log_transform(trgt, self.log_transform_loss_shift)

                loss = F.l1_loss(pred, trgt, reduction='none') * self.loss_weight

                if self.split_loss == 'none':
                    losses[key] = loss[mask_observed | mask_outside].mean()

                elif self.split_loss == 'pred':
                    if i == 0:
                        # no sparsifing mask for first resolution
                        losses[key] = loss[mask_observed | mask_outside].mean()
                    else:
                        mask = mask_surface_pred[i - 1] & (mask_observed | mask_outside)
                        if mask.sum() > 0:
                            losses[key] = loss[mask].mean()
                        else:
                            losses[key] = 0 * loss.sum()

                else:
                    raise NotImplementedError("TSDF loss split [%s] not supported" % self.split_loss_empty)

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

    def forward(self, xs, targets=None):
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
                trgt = targets[key]
                mask_surface = targets['vol_%02d_tsdf'%voxel_size].squeeze(1).abs() < 1

                loss = F.cross_entropy(pred, trgt, reduction='none', ignore_index=-1)
                if mask_surface.sum()>0 and  not loss.isnan().any():
                    loss = loss[mask_surface].mean()
                else:
                    loss = 0 * loss.mean()
                losses[key] = loss * self.loss_weight

        return output, losses


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
    def __init__(self, alpha=0.25, gamma=2, thres=0.0, logits=False, reduce=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.thres = thres

    def forward(self, inputs, targets):
        pos_mask = (targets >= self.thres).float()
        F_loss = -pos_mask *self.alpha*((1.-inputs)**self.gamma)*torch.log(inputs+1e-6) \
                               -(1.-pos_mask)*(1.-self.alpha)*(inputs** self.gamma)*torch.log(1.-inputs+1e-6)
#
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
#
# class CenProbHead(nn.Module):
#     """ Predicts voxel semantic segmentation"""
#
#     def __init__(self, cfg):
#         super().__init__()
#
#         self.pos_thres = cfg.MODEL.GROUPING.PROB_THRES
#         self.multi_scale = cfg.MODEL.HEADS3D.MULTI_SCALE
#         self.loss_weight = cfg.MODEL.HEADS3D.CENPROB.LOSS_WEIGHT
#
#         scales = len(cfg.MODEL.BACKBONE3D.CHANNELS)-1
#         final_size = int(cfg.VOXEL_SIZE*100)
#         norm = cfg.MODEL.BACKBONE3D.NORM
#
#         if self.multi_scale:
#             self.voxel_sizes = [final_size*2**i for i in range(scales)][::-1]
#
#             pre_convs, decoders = [], []
#             xy_pool_convs, yz_pool_convs, xz_pool_convs = [], [] , []
#
#             for c in cfg.MODEL.BACKBONE3D.CHANNELS[:-1][::-1]:
#                 out_c = c//4
#                 pre_convs.append(BasicBlock3d(c, out_c, dilation=1, norm=norm))
#                 xy_pool_convs.append(nn.ModuleList([conv3dNorm(out_c, out_c, norm, 1, 1, 0), # output 2*2*2
#                                                    conv3dNorm(out_c, out_c, norm, 1, 1, 0),
#                                                    conv3dNorm(out_c, out_c, norm, 1, 1, 0),
#                                                    conv3dNorm(out_c, out_c, norm, 1, 1, 0)]))
#                 yz_pool_convs.append(nn.ModuleList([conv3dNorm(out_c, out_c, norm, 1, 1, 0), # output 2*2*2
#                                                    conv3dNorm(out_c, out_c, norm, 1, 1, 0),
#                                                    conv3dNorm(out_c, out_c, norm, 1, 1, 0),
#                                                    conv3dNorm(out_c, out_c, norm, 1, 1, 0)]))
#                 xz_pool_convs.append(nn.ModuleList([conv3dNorm(out_c, out_c, norm, 1, 1, 0), # output 2*2*2
#                                                    conv3dNorm(out_c, out_c, norm, 1, 1, 0),
#                                                    conv3dNorm(out_c, out_c, norm, 1, 1, 0),
#                                                    conv3dNorm(out_c, out_c, norm, 1, 1, 0)]))
#                 decoders.append(nn.Conv3d(c, 1, 1, bias=False) )
#
#         self.pre_convs, self.decoders = nn.ModuleList(pre_convs),nn.ModuleList(decoders) # the default precision is float32, while if train with 16, have to use this
#         self.xy_pool_convs, self.yz_pool_convs, self.xz_pool_convs = nn.ModuleList(xy_pool_convs),  \
#                                                                      nn.ModuleList(yz_pool_convs),  nn.ModuleList(xz_pool_convs),
#         self.loss = ShrinkageLoss(a=2., c=0.2,
#                                   reduce=False)  # FocalLoss(alpha=0.25, gamma=2, thres=self.pos_thres, reduce=False)
#
#     def forward(self, xs, targets=None):
#         output = {}
#         losses = {}
#
#         if not self.multi_scale:
#             xs = xs[-1:] # just use final scale
#
#
#         for voxel_size, pre_convs, xy_pool_conv, yz_pool_conv, xz_pool_conv, decoder, x in \
#                 zip(self.voxel_sizes, self.pre_convs, self.xy_pool_convs, self.yz_pool_convs,self.xz_pool_convs, self.decoders, xs):
#             # compute semantic labels
#             key = 'vol_%02d_centroid_prob'%voxel_size
#             x = pre_convs(x)
#
#             # 3 view pooling
#             b, c, d, h, w = x.shape
#             xy_view, yz_view, xz_view = x.clone(), x.clone(), x.clone()
#             for i, pool_size in enumerate(np.linspace(2, d // 2, 4, dtype=int)):
#
#                 # overcome the same running_mean == nan issue mentioned in backbone3d basic block
#                 # if self.training and (xy_pool_conv[i][1].running_mean.isnan().any() or xy_pool_conv[i][1].running_var.isnan().any()):
#                 #     # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
#                 #     print('running_stats reset')
#                 #     xy_pool_conv[i][1].running_mean.zero_()
#                 #     xy_pool_conv[i][1].running_var.fill_(1)
#                 #     xy_pool_conv[i][1].num_batches_tracked.zero_()
#
#                 kernel_size = (int(d / pool_size), 1, 1)
#                 out = F.max_pool3d(x, kernel_size, stride=kernel_size)
#                 out = xy_pool_conv[i](out)
#                 out = F.interpolate(out, size=(d, h, w), mode='nearest')
#                 xy_view += 0.25*out
#             xy_view = F.relu(xy_view / 2., inplace=True)
#
#             for i, pool_size in enumerate(np.linspace(2,h // 2, 4, dtype=int)):
#
#                 # if self.training and (xz_pool_conv[i][1].running_mean.isnan().any()  or xz_pool_conv[i][1].running_var.isnan().any() ):
#                 #     # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
#                 #     print('running_stats reset')
#                 #     xz_pool_conv[i][1].running_mean.zero_()
#                 #     xz_pool_conv[i][1].running_var.fill_(1)
#                 #     xz_pool_conv[i][1].num_batches_tracked.zero_()
#
#                 kernel_size = (1, int(h / pool_size), 1)
#                 out = F.max_pool3d(x, kernel_size, stride=kernel_size)
#                 out = xz_pool_conv[i](out)
#                 out = F.interpolate(out, size=(d, h, w), mode='nearest')
#                 xz_view += 0.25*out
#             xz_view = F.relu(xz_view / 2., inplace=True)
#
#             for i, pool_size in enumerate(np.linspace(2, w // 2, 4, dtype=int)):
#
#                 # if self.training and (yz_pool_conv[i][1].running_mean.isnan().any() or yz_pool_conv[i][1].running_var.isnan().any() ):
#                 #     # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
#                 #     print('running_stats reset')
#                 #     yz_pool_conv[i][1].running_mean.zero_()
#                 #     yz_pool_conv[i][1].running_var.fill_(1)
#                 #     yz_pool_conv[i][1].num_batches_tracked.zero_()
#
#                 kernel_size = (1, 1, int(w / pool_size))
#                 out = F.max_pool3d(x, kernel_size, stride=kernel_size)
#                 out = yz_pool_conv[i](out)
#                 out = F.interpolate(out, size=(d, h, w), mode='nearest')
#                 yz_view += 0.25*out
#             yz_view = F.relu(yz_view / 2., inplace=True)
#
#             vol = torch.cat([x, xy_view, xz_view, yz_view], dim=1)
#             output[key] = torch.sigmoid(decoder(vol)) # b*1*nx*ny*nz torch.sigmoid(
#
#             # compute losses
#             if targets is not None and key in targets:
#                 pred = output[key]
#                 trgt = targets[key].unsqueeze(1)
#                 mask_surface = targets['vol_%02d_tsdf' % voxel_size].abs() < 1
#                 # mask_vaild = (trgt >= 0)
#                 # mask = mask_surface #& mask_vaild
#                 # pos_w = torch.as_tensor(2.)# mask.sum().type(torch.float)/ ((trgt > self.pos_thres).sum().type(torch.float))
#                 # print("center shape {}, {}, mask {}".format(pred.shape, trgt.shape, mask.shape))
#                 # print("max val {}, {}".format(pred[mask].max(), trgt[mask].max()))
#
#                 loss = self.loss(pred, trgt)
#                 if mask_surface.sum() > 0 and not loss.isnan().any():
#                     loss = loss[mask_surface].mean()
#                 else:
#                     loss = 0 * loss.mean()
#
#                 # loss = F.binary_cross_entropy_with_logits(pred, trgt, reduction='none', pos_weight=pos_w)
#                 #
#                 # if mask.sum() > 0:
#                 #     loss = loss[mask].mean()
#                 # else:
#                 #     loss = 0 *  loss.mean()
#                 #
#                 losses[key] = loss * self.loss_weight
#
#             output[key] = output[key] #torch.sigmoid(output[key]) #
#
#
#         return output, losses
#
#
#
#
#
#
# class ColorHead(nn.Module):
#     """ Predicts voxel color"""
#
#     def __init__(self, cfg):
#         super().__init__()
#
#         self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1, 1)
#         self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1, 1)
#
#         self.multi_scale = cfg.MODEL.HEADS3D.MULTI_SCALE
#         self.loss_weight = cfg.MODEL.HEADS3D.COLOR.LOSS_WEIGHT
#
#         scales = len(cfg.MODEL.BACKBONE3D.CHANNELS)-1
#         final_size = int(cfg.VOXEL_SIZE*100)
#
#         if self.multi_scale:
#             self.voxel_sizes = [final_size*2**i for i in range(scales)][::-1]
#             decoders = [nn.Conv3d(c, 3, 1, bias=False)
#                         for c in cfg.MODEL.BACKBONE3D.CHANNELS[:-1]][::-1]
#         else:
#             self.voxel_sizes = [final_size]
#             decoders = [nn.Conv3d(cfg.MODEL.BACKBONE3D.CHANNELS[0], 3, 1, bias=False)]
#
#         self.decoders = nn.ModuleList(decoders)
#
#     def forward(self, xs, targets=None):
#         output = {}
#         losses = {}
#
#         if not self.multi_scale:
#             xs = xs[-1:] # just use final scale
#
#         for voxel_size, decoder, x in zip(self.voxel_sizes, self.decoders, xs):
#             key = 'vol_%02d_color'%voxel_size
#             pred = torch.sigmoid(decoder(x)) * 255
#             output[key] = pred
#
#             # compute losses
#             if targets is not None and key in targets:
#                 pred = output[key]
#                 trgt = targets[key]
#                 mask_surface = targets['vol_%02d_tsdf'%voxel_size].squeeze(1).abs() < 1
#
#                 loss = F.l1_loss(pred, trgt, reduction='none').mean(1)
#                 if mask_surface.sum()>0:
#                     loss = loss[mask_surface].mean()
#                 else:
#                     loss = 0 * loss.mean()
#                 losses[key] = loss * self.loss_weight / 255
#
#         return output, losses
#
#
#
#
