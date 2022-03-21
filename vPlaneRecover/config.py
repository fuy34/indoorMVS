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
#  Last modification: Fengting Yang, 03/22/2022

import argparse
from fvcore.common.config import CfgNode as _CfgNode


def convert_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, _CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict 

class CfgNode(_CfgNode):
    """Remove once  https://github.com/rbgirshick/yacs/issues/19 is merged"""
    def convert_to_dict(self):
        return convert_to_dict(self)

CN = CfgNode


_C = CN()
_C.LOG_DIR = "/data/Fengting/vPlaneRecover_train/"
_C.RESUME_CKPT = None #'/data/ScanNet/vPlaneRecover_train/vPlaneRecover/3heads_planeCls_coordCov_branchRelated_ctrw1_server/epoch=049.ckpt'
_C.VOXEL_SIZE = .04
_C.VOXEL_DIM_TRAIN = [160,160,64]
_C.VOXEL_DIM_VAL = [256,256,96] # this is the dim of vol_04 - volume with voxel size 4cm

_C.DATASETS_TRAIN = ['meta_file/scannet_train_demo.txt']
_C.DATASETS_VAL = ['meta_file/scannet_val_demo.txt'] #_debug3  _debug_server

_C.DATA = CN()
_C.DATA.BATCH_SIZE_TRAIN = 1 # batch size per gpu
_C.DATA.NUM_FRAMES_TRAIN = 50 # memory run out of directly in 2D backbone if we choose 5
_C.DATA.NUM_FRAMES_VAL = 100
_C.DATA.FRAME_SELECTION = 'random'
_C.DATA.RANDOM_ROTATION_3D = True
_C.DATA.RANDOM_TRANSLATION_3D = True
_C.DATA.PAD_XY_3D = 1.
_C.DATA.PAD_Z_3D = .25

_C.TRAINER = CN()
_C.TRAINER.NAME = "indoorMVS"
_C.TRAINER.VERSION = "HT"
_C.TRAINER.NUM_GPUS = 1
_C.TRAINER.NUM_DATALOADER = 4 # max for our server due to the memory bank
_C.TRAINER.PRECISION = 16 # 16bit of 32bit
_C.TRAINER.BATCH_BACKBONE2D_TIME = True
_C.TRAINER.FINETUNE3D = False
_C.TRAINER.CHECKPOINT_PERIOD = 5
_C.TRAINER.NUM_EPOCH = 150

_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = "Adam"
_C.OPTIMIZER.ADAM = CN()
_C.OPTIMIZER.ADAM.LR = 5e-4
_C.OPTIMIZER.BACKBONE2D_LR_FACTOR = 1.
_C.SCHEDULER = CN()
_C.SCHEDULER.TYPE = "StepLR"
_C.SCHEDULER.STEP_LR = CN()
_C.SCHEDULER.STEP_LR.STEP_SIZE = 150 #200
_C.SCHEDULER.STEP_LR.GAMMA = .1


_C.MODEL = CN()
#TODO: images are currently loaded RGB, but the pretrained models expect BGR
_C.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
_C.MODEL.PIXEL_STD = [1., 1., 1.]

# for custom resnet18-fpn
_C.MODEL.RESNETS = CN()
_C.MODEL.RESNETS.TRAINABLE_LAYERS= 3
_C.MODEL.RESNETS.DEPTH = 18 # 50
_C.MODEL.RESNETS.PRETRAIN=True

_C.MODEL.FPN = CN()
_C.MODEL.FPN.NORM = "nnSyncBN"
_C.MODEL.FPN.FUSE_TYPE = "sum"


_C.MODEL.BACKBONE3D = CN()
_C.MODEL.BACKBONE3D.CHANNELS = [32, 64, 128, 256]
_C.MODEL.BACKBONE3D.LAYERS_DOWN = [1, 2, 3, 4] # the number of 3D conv Block in encoder layers
_C.MODEL.BACKBONE3D.LAYERS =[3, 2, 1] # the number of 3D conv block in decoder layer
_C.MODEL.BACKBONE3D.NORM = "nnSyncBN" #'GN' #
_C.MODEL.BACKBONE3D.DROP = 0.
_C.MODEL.BACKBONE3D.CONDITIONAL_SKIP = False
_C.MODEL.BACKBONE3D.THETA_STEP = 90
_C.MODEL.BACKBONE3D.PHI_STEP = 5
_C.MODEL.BACKBONE3D.RHO_STEP = [1, 2, 4]

_C.MODEL.HEADS2D = CN()
_C.MODEL.HEADS2D.HEADS = [""]
_C.MODEL.HEADS2D.SEMSEG = CN()
_C.MODEL.HEADS2D.SEMSEG.NUM_CLASSES = 41
_C.MODEL.HEADS2D.SEMSEG.LOSS_WEIGHT = 1.

_C.MODEL.HEADS3D = CN()
_C.MODEL.HEADS3D.HEADS = ['tsdf', 'semseg', 'plane_ins']
_C.MODEL.HEADS3D.MULTI_SCALE = True  # Fixed to be true
_C.MODEL.HEADS3D.TSDF = CN()
_C.MODEL.HEADS3D.TSDF.LOSS_WEIGHT = 1.
_C.MODEL.HEADS3D.TSDF.LABEL_SMOOTHING = 1.05
_C.MODEL.HEADS3D.TSDF.LOSS_SPLIT = 'pred'
_C.MODEL.HEADS3D.TSDF.LOSS_LOG_TRANSFORM = True
_C.MODEL.HEADS3D.TSDF.LOSS_LOG_TRANSFORM_SHIFT = 1.0
_C.MODEL.HEADS3D.TSDF.SPARSE_THRESHOLD = [.99, .99, .99]
_C.MODEL.HEADS3D.SEMSEG = CN()
_C.MODEL.HEADS3D.SEMSEG.NUM_CLASSES = 41
_C.MODEL.HEADS3D.SEMSEG.LOSS_WEIGHT = .05
_C.MODEL.HEADS3D.COLOR = CN()
_C.MODEL.HEADS3D.COLOR.LOSS_WEIGHT = 1.
_C.MODEL.HEADS3D.HTMAP = CN()
_C.MODEL.HEADS3D.HTMAP.LOSS_WEIGHT = 40.

_C.MODEL.GROUPING = CN()
_C.MODEL.GROUPING.PROB_THRES = 0.001
_C.MODEL.GROUPING.NORM_THRES = 30 #degree
# _C.MODEL.GROUPING.TOPK_PROB = 4000


def get_parser():
    parser = argparse.ArgumentParser(description="IndoorMVS Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def get_cfg(args):
    cfg = _C.clone()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

