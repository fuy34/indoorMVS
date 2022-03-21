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
import torch.nn as nn
import torch.nn.functional as F
from vPlaneRecover.model_util import *
from vPlaneRecover.planeHT import Plane_HT,  load_vote_idx

from functools import reduce



class ConditionalProjection(nn.Module):
    """ Applies a projected skip connection from the encoder to the decoder

    When condition is False this is a standard projected skip connection
    (conv-bn-relu).

    When condition is True we only skip the non-masked features
    from the encoder. To maintin scale we instead skip the decoder features.
    This was intended to reduce artifacts in unobserved regions,
    but was found to not be helpful.
    """

    def __init__(self, n, norm='BN', condition=True):
        super(ConditionalProjection, self).__init__()
        # return relu(bn(conv(x)) if mask, relu(bn(y)) otherwise
        self.conv = conv1x1x1(n, n)
        self.norm = get_norm_3d(norm, n)
        self.relu = nn.ReLU(True)
        self.condition = condition

    def forward(self, x, y, mask):
        """
        Args:
            x: tensor from encoder
            y: tensor from decoder
            mask
        """

        x = self.conv(x)
        if self.condition:
            x = torch.where(mask, x, y)
        x = self.norm(x)
        x = self.relu(x)
        return x


class EncoderDecoder(nn.Module):
    """ 3D network to refine feature volumes"""

    def __init__(self,  voxel_dim_train = [160, 160, 64], voxel_dim_val = [256,256,128],
                 channels=[32,64,128], layers_down=[1,2,3], layers_up=[3,3,3],
                 norm='BN', drop=0, zero_init_residual=True,
                 cond_proj=False, phi_step=5, theta_step=90, rho_step=[1, 1, 2], do_HT=True):
        super(EncoderDecoder, self).__init__()

        # for plane HT
        self.voxel_dim_train = voxel_dim_train
        self.voxel_dim_val = voxel_dim_val
        self.ht_neck = nn.ModuleList()
        self.vote_idxs = []
        self.phi_step, self.theta_step, self.rho_step = phi_step, theta_step, rho_step

        # planeHT
        # self.init_vote_idx(voxel_dim_train, phi_step, theta_step, rho_step)
        self.do_HT = do_HT
        self.vote_idxs = []
        self.phi_step, self.theta_step, self.rho_step = phi_step, theta_step, rho_step

        # input norm
        self.input_norm = get_norm_3d(norm, channels[0])

        # Atlas
        self.cond_proj = cond_proj

        self.layers_down = nn.ModuleList()
        self.proj = nn.ModuleList()

        self.layers_down.append(nn.Sequential(*[
            BasicBlock3d(channels[0], channels[0], norm=norm, drop=drop) 
            for _ in range(layers_down[0]) ]))
        self.proj.append( ConditionalProjection(channels[0], norm, cond_proj) )
        for i in range(1,len(channels)):
            layer = [nn.Conv3d(channels[i-1], channels[i], 3, 2, 1, bias=(norm=='')),
                     get_norm_3d(norm, channels[i]),
                     nn.Dropout(drop, True),
                     nn.ReLU(inplace=True)]
            layer += [BasicBlock3d(channels[i], channels[i], norm=norm, drop=drop) 
                      for _ in range(layers_down[i])]
            self.layers_down.append(nn.Sequential(*layer))
            if i<len(channels)-1:
                self.proj.append( ConditionalProjection(channels[i], norm, cond_proj) )

        self.proj = self.proj[::-1]

        channels = channels[::-1] #[256 128 64 32]
        self.layers_up_conv = nn.ModuleList()
        self.layers_up_res = nn.ModuleList()
        # self.mergeFeat = nn.ModuleList()
        for i in range(1,len(channels)):
            if i < len(channels):
                # neck only applied to 8 16 voxel_sz
                self.ht_neck.append(Plane_HT(channels[i], norm, drop))
                # self.mergeFeat.append(conv1x1x1(channels[i]*2, channels[i]))

            self.layers_up_conv.append( conv1x1x1(channels[i-1], channels[i]) )
            self.layers_up_res.append(nn.Sequential( *[
                BasicBlock3d(channels[i], channels[i], norm=norm, drop=drop) 
                for _ in range(layers_up[i-1]) ]))

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each 
        # residual block behaves like an identity. This improves the 
        # model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock3d):
                    nn.init.constant_(m.bn2.weight, 0)


    def forward(self, x):
        if self.cond_proj: # False
            valid_mask = (x!=0).any(1, keepdim=True).float()

        # reinit vote idx if changed from train to val
        if self.do_HT:
            self.check_vote_idx(x)


        # each layer has 1 Conv3D + n Block3d
        xs = []
        for layer in self.layers_down:
            x = layer(x)
            xs.append(x)

        xs = xs[::-1]
        out, out_htmap = [], []
        for i in range(len(self.layers_up_conv)):
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
            x = self.layers_up_conv[i](x)
            if self.cond_proj:
                scale = 1/2**(len(self.layers_up_conv)-i-1)
                mask = F.interpolate(valid_mask, scale_factor=scale)!=0 
            else:
                mask = None
            y = self.proj[i](xs[i+1], x, mask)
            x = (x + y)/2  # skip connect with avg
            x = self.layers_up_res[i](x)

            # transfer voxel_sz 8 16 feat to HT space
            if i < len(self.layers_up_conv) - 1 and self.do_HT:
                HT_att, HT_htmap = self.ht_neck[i](x, self.vote_idxs[i], self.rhos[i], self.thetas[i], self.phis[i])
            else:
                HT_htmap = None


            out_htmap.append(HT_htmap)
            out.append(x) # 1/4, 1/2, 1

        return out, out_htmap

    def init_vote_idx(self, voxel_dim, phi_step, theta_step, rho_step):
        self.vote_idxs = []
        self.rhos, self.thetas, self.phis, self.diag2 = [], [], [], []
        self.n_rho, self.n_tht, self.n_phi = [], [], []
        for i in range(1, len(rho_step)):
            # we use 1/2, 1/4, 1/8
            vote_idx, rhos, thetas, phis, diag = load_vote_idx([x // 2 ** (len(rho_step) - i) for x in voxel_dim],
                                                               phi_step, theta_step, rho_step[i])
            self.vote_idxs.append(vote_idx.float())  # note pytorch  sparse.mm only support float type
            self.rhos.append(rhos)
            self.thetas.append(thetas)
            self.phis.append(phis)

            self.n_rho.append(len(rhos))
            self.n_tht.append(len(thetas))
            self.n_phi.append(len(phis))

    def check_vote_idx(self, x):
        n = len(self.vote_idxs)
        if self.training:
            if len(self.vote_idxs) > 0:
                b_right = True
                for i, idx_map in enumerate(self.vote_idxs):
                    b_right &= (idx_map.shape[1] == \
                                reduce((lambda x, y: x * y), self.voxel_dim_train) / (
                                            (2 ** len(self.voxel_dim_val)) ** (n - i)))
            else:
                b_right = False

            if not b_right:
                if len(self.vote_idxs) > 0:
                    print(self.vote_idxs[0].shape)
                print('re-init vote_idx map (val-->train)')
                self.init_vote_idx(self.voxel_dim_train, self.phi_step, self.theta_step, self.rho_step)

            if self.vote_idxs[0].device != x.device:
                for i in range(len(self.vote_idxs)):
                    self.vote_idxs[i] = self.vote_idxs[i].to(x.device)
        else:
            if len(self.vote_idxs) > 0:
                b_right = True
                for i, idx_map in enumerate(self.vote_idxs):
                    b_right &= (idx_map.shape[1] == \
                                reduce((lambda x, y: x * y), self.voxel_dim_val) / (
                                            (2 ** len(self.voxel_dim_val)) ** (n - i)))
            else:
                b_right = False

            if not b_right:
                if len(self.vote_idxs) > 0:
                    print(self.vote_idxs[0].shape)
                print('re-init vote_idx map (train --> val')
                self.init_vote_idx(self.voxel_dim_val, self.phi_step, self.theta_step, self.rho_step)

            if self.vote_idxs[0].device != x.device:
                for i in range(len(self.vote_idxs)):
                    self.vote_idxs[i] = self.vote_idxs[i].to(x.device)

def build_backbone3d(cfg):
    return EncoderDecoder(
        cfg.VOXEL_DIM_TRAIN, cfg.VOXEL_DIM_VAL,
        cfg.MODEL.BACKBONE3D.CHANNELS, cfg.MODEL.BACKBONE3D.LAYERS_DOWN,
        cfg.MODEL.BACKBONE3D.LAYERS, cfg.MODEL.BACKBONE3D.NORM,
        cfg.MODEL.BACKBONE3D.DROP, True, cfg.MODEL.BACKBONE3D.CONDITIONAL_SKIP,
         cfg.MODEL.BACKBONE3D.PHI_STEP, cfg.MODEL.BACKBONE3D.THETA_STEP, cfg.MODEL.BACKBONE3D.RHO_STEP,
        do_HT = 'plane_ins' in cfg.MODEL.HEADS3D.HEADS
    )



