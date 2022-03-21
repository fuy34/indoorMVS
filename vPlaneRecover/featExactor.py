from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
# import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size=3,  stride=1, padding=1, bias=False,  norm= nn.SyncBatchNorm):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        self.cbr_unit = nn.Sequential(conv_mod, norm(n_filters), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class pyramidPooling(nn.Module):
    # adopted from https://github.com/gengshan-y/high-res-stereo/blob/aae0b9b86c4ab007f83ed0f583f9ed7ff4b032ea/models/utils.py
    def __init__(self, in_channels, pool_sizes, norm=nn.SyncBatchNorm, model_name='pspnet', fusion_mode='cat', with_bn=True):
        super(pyramidPooling, self).__init__()

        bias = not with_bn

        self.paths = []
        if pool_sizes is None:
            for i in range(4):
                self.paths.append(conv2DBatchNormRelu(in_channels, in_channels, 1, 1, 0, bias=bias, norm=norm))
        else:
            for i in range(len(pool_sizes)):
                self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=bias, norm=norm))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode

    #@profile
    def forward(self, x):
        h, w = x.shape[2:]

        k_sizes = []
        strides = []
        if self.pool_sizes is None:
            for pool_size in np.linspace(1,min(h,w)//2,4,dtype=int):
                k_sizes.append((int(h/pool_size), int(w/pool_size)))
                strides.append((int(h/pool_size), int(w/pool_size)))
            k_sizes = k_sizes[::-1]
            strides = strides[::-1]
        else:
            k_sizes = [(self.pool_sizes[0],self.pool_sizes[0]),(self.pool_sizes[1],self.pool_sizes[1]) ,
                       (self.pool_sizes[2],self.pool_sizes[2]) ,(self.pool_sizes[3],self.pool_sizes[3])]
            strides = k_sizes

        if self.fusion_mode == 'cat': # pspnet: concat (including x)
            output_slices = [x]

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                #out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != 'icnet':
                    out = module(out)
                out = F.upsample(out, size=(h,w), mode='bilinear')
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else: # icnet: element-wise sum (including x)
            pp_sum = x.clone()

            for i, module in enumerate(self.path_module_list):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                out = module(out)
                out = F.upsample(out, size=(h,w), mode='bilinear')
                pp_sum = pp_sum + 0.25*out
            pp_sum = F.relu(pp_sum/2.,inplace=True)

            return pp_sum



class FeatExactor(nn.Module):
    """Pytorch module for a resnet encoder
     Adopted from https://github.com/nianticlabs/monodepth2/blob/master/networks/resnet_encoder.py
     https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
     https://github.com/pytorch/vision/blob/master/torchvision/models/detection/backbone_utils.py
     https://github.com/gengshan-y/high-res-stereo/blob/master/models/utils.py
    """
    def __init__(self, num_layers=18, out_channel=32, norm='nnSyncBN',  pretrained=True, trainable_layers=3):
        super(FeatExactor, self).__init__()

        if num_layers not in [18, 34, 50]:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50}

        BN = {
            "BN": nn.BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]

        self.freezon_param = []
        # backbone = resnet_fpn_backbone('resnet{}'.format(num_layers),  pretrained, norm_layer=BN, trainable_layers=)
        self.encoder = resnets[num_layers](pretrained, norm_layer=BN)

        layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
        # freeze layers only if pretrained backbone is used
        for name, parameter in self.encoder.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
                self.freezon_param += list(parameter)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        # self.output_stride = [4, 8, 16, 32]

        self.pyramid_pooling = pyramidPooling( self.num_ch_enc[-1], None, fusion_mode='sum', model_name='icnet', norm=BN)
        # Iconvs
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                        conv2DBatchNormRelu(in_channels= self.num_ch_enc[-1], k_size=3, n_filters= self.num_ch_enc[-2], norm=BN))
        self.iconv5 = conv2DBatchNormRelu(in_channels= self.num_ch_enc[-1], k_size=3, n_filters=self.num_ch_enc[-2], norm=BN)

        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                         conv2DBatchNormRelu(in_channels= self.num_ch_enc[-2], k_size=3, n_filters= self.num_ch_enc[-3],norm=BN))
        self.iconv4 = conv2DBatchNormRelu(in_channels=self.num_ch_enc[-2], k_size=3, n_filters= self.num_ch_enc[-3], padding=1, stride=1, norm=BN)

        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels= self.num_ch_enc[-3], k_size=3, n_filters= self.num_ch_enc[-4], norm=BN))
        self.iconv3 = conv2DBatchNormRelu(in_channels= self.num_ch_enc[-3], k_size=3, n_filters= self.num_ch_enc[-4], norm=BN)

        self.proj6 = conv2DBatchNormRelu(in_channels= self.num_ch_enc[-1], k_size=1, n_filters=out_channel, padding=0, stride=1,norm=BN)
        self.proj5 = conv2DBatchNormRelu(in_channels= self.num_ch_enc[-2], k_size=1, n_filters=out_channel, padding=0, stride=1, norm=BN)
        self.proj4 = conv2DBatchNormRelu(in_channels= self.num_ch_enc[-3], k_size=1, n_filters=out_channel, padding=0, stride=1, norm=BN)
        self.proj3 = conv2DBatchNormRelu(in_channels= self.num_ch_enc[-4], k_size=1, n_filters=out_channel, padding=0, stride=1, norm=BN)

    def forward(self, input_image):
        self.features = []
        x = input_image.clone()

        # H, W -> H/2, W/2
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)

        ## H/2, W/2 -> H/4, W/4
        pool1 = self.encoder.maxpool(x)

        # H/4, W/4 -> H/16, W/16
        conv3 = self.encoder.layer1(pool1)
        conv4 = self.encoder.layer2(conv3)
        conv5 = self.encoder.layer3(conv4)
        conv6 = self.encoder.layer4(conv5)
        conv6 = self.pyramid_pooling(conv6)


        concat5 = torch.cat((conv5, self.upconv6(conv6)), dim=1)
        conv5 = self.iconv5(concat5)

        concat4 = torch.cat((conv4, self.upconv5(conv5)), dim=1)
        conv4 = self.iconv4(concat4)

        concat3 = torch.cat((conv3, self.upconv4(conv4)), dim=1)
        conv3 = self.iconv3(concat3)

        proj6 = F.interpolate(self.proj6(conv6),scale_factor=2, mode='bilinear', align_corners=False)
        proj5 = F.interpolate(self.proj5(conv5) + proj6, scale_factor=2, mode='bilinear', align_corners=False)
        proj4 = F.interpolate(self.proj4(conv4) + proj5, scale_factor=2, mode='bilinear', align_corners=False)
        proj3 = self.proj3(conv3) + proj4

        return proj3 #, proj4, proj5, proj6 #1/4, 1/8, 1/16, 1/32


def build_backbone2d(cfg):
    """ Builds 2D feature extractor backbone network from Detectron2."""

    output_dim = cfg.MODEL.BACKBONE3D.CHANNELS[0]
    output_stride = 4
    feature_extractor = FeatExactor(cfg.MODEL.RESNETS.DEPTH, out_channel=output_dim, norm= cfg.MODEL.FPN.NORM,
                                    pretrained=cfg.MODEL.RESNETS.PRETRAIN,
                                    trainable_layers=cfg.MODEL.RESNETS.TRAINABLE_LAYERS)

    return feature_extractor, output_stride

