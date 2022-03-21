import torch.nn as nn
import torch
import numpy as np

import os

def get_norm_3d(norm, out_channels):
    """ Get a normalization module for 3D tensors

    Args:
        norm: (str or callable)
        out_channels

    Returns:
        nn.Module or None: the normalization layer
    """

    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm3d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm(out_channels)

def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)

class conv3dNorm(nn.Module):
    def __init__(self, inchannel, outChannel, norm = 'BN', kernel=1, stride=1, pad=0):
        super(conv3dNorm, self).__init__()
        self.conv = nn.Conv3d(inchannel, outChannel, kernel_size=kernel, padding=pad, stride=stride, bias=False)
        self.bn = get_norm_3d(norm,outChannel)

    def forward(self, x):
        if self.training and (self.bn.running_mean.isnan().any()  or self.bn.running_var.isnan().any()):
            # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
            print('running_stats reset')
            self.bn.running_mean.zero_()
            self.bn.running_var.fill_(1)
            self.bn.num_batches_tracked.zero_()

        return self.bn(self.conv(x))



class BasicBlock3d(nn.Module):
    """ 3x3x3 Resnet Basic Block"""
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None,dilation=1, norm='BN', drop=0):
        super(BasicBlock3d, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3x3(inplanes, planes, stride, 1, dilation)
        self.bn1 = get_norm_3d(norm, planes)
        # self.drop1 = nn.Dropout(drop, True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, 1, 1, dilation)
        self.bn2 = get_norm_3d(norm, planes)
        # self.drop2 = nn.Dropout(drop, True)
        self.norm = norm
        if downsample is not None:
            self.downsample = downsample
        elif stride != 1 or inplanes != planes:
            # from https://github.com/kenshohara/3D-ResNets-PyTorch/blob/8e6a026d57eda8eb54db45090e315c310750762f/models/resnet.py
            self.downsample = nn.Sequential(conv1x1x1(inplanes, planes, stride),  get_norm_3d(norm, planes))
        else:
            self.downsample = None

        self.stride = stride

    def forward(self, x):
        # note, if there is x are all same in one channel,  the var == 0 and then running_var will be 0
        # the running_mean and var will be nan
        # https://github.com/pytorch/pytorch/issues/1206#issuecomment-292440241
        # https://discuss.pytorch.org/t/nan-when-i-use-batch-normalization-batchnorm1d/322/9
        # https://discuss.pytorch.org/t/solved-get-nan-for-input-after-applying-a-convolutional-layer/8165/39
        # it very rarelly happends, but could be in our projection case, we reset the runnin stats in this case
        if self.training and (self.bn1.running_mean.isnan().any()  or self.bn1.running_var.isnan().any()):
            # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
            print('running_stats reset')
            self.bn1.running_mean.zero_()
            self.bn1.running_var.fill_(1)
            self.bn1.num_batches_tracked.zero_()

        if self.training and (self.bn2.running_mean.isnan().any()  or self.bn2.running_var.isnan().any()) :
            # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
            print('running_stats reset')
            self.bn2.running_mean.zero_()
            self.bn2.running_var.fill_(1)
            self.bn2.num_batches_tracked.zero_()

        if self.training and (self.downsample is not None) and\
                (self.downsample[1].running_var.isnan().any() or self.downsample[1].running_mean.isnan().any()):
                print('running_stats reset')
                self.downsample[1].running_mean.zero_()
                self.downsample[1].running_var.fill_(1)
                self.downsample[1].num_batches_tracked.zero_()

        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        # out = self.bn1(out) #if self.training and (torch.var(out,dim=(0,2,3,4)) == 0).sum() == 0 else out - out.mean(dim=(0,2,3,4),keepdim=True)
      # out = self.drop1(out) # drop after both??
        out = self.relu(out)

        out = self.bn2(out)
        out = self.conv2(out)
        # out = self.bn2(out) #if self.training and (torch.var(out,dim=(0,2,3,4)) == 0).sum() == 0 else out - out.mean(dim=(0,2,3,4),keepdim=True)
        # out = self.drop2(out) # drop after both??

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)


        return out

