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

import os

from pytorch_lightning.loggers import TensorBoardLogger
import torch


class MeshWriter:
    """ Saves mesh to logdir during training"""

    def __init__(self, save_path):
        self._save_path = os.path.join(save_path, "train_viz")
        if not os.path.isdir(self._save_path):
            os.makedirs(self._save_path, exist_ok=True)
  
    def save_mesh(self, tsdf, name):
        output_meshs = []
        if 'semseg' in tsdf.attribute_vols:
            output_meshs.append('semseg')

        if 'centroid_prob' in tsdf.attribute_vols:
            output_meshs.append('centroid_prob')

        if 'plane_ins' in tsdf.attribute_vols:
            output_meshs.append('plane_ins')
            output_meshs.append('vert_plane')

        if 'plane_norm' in tsdf.attribute_vols:
            output_meshs.append('normal')

        if 'plane_cls' in tsdf.attribute_vols:
            output_meshs.append('plane_cls')

        if 'param_htmap' in tsdf.attribute_vols:
            output_meshs.append('param_htmap')

        meshes = tsdf.get_mesh(output_meshs)

        pth = os.path.join(self._save_path, name)
        if not os.path.isdir(pth):
            os.makedirs(pth)

        if isinstance(meshes, dict):
            for key in meshes:
                if key == 'semseg':
                    meshes[key].export(pth + '/' + name + '_semseg.ply')
                else:
                    # print(key, meshes[key])
                    meshes[key].export(pth + '/' + name + '_{}.ply'.format(key))
        else: # fail mc and get trimesh
            meshes.export(pth + '/' + name + '_empty.ply')



class AtlasLogger(TensorBoardLogger):
    """ Does tensorboard logging + has a MeshWriter for saving example
    meshes throughout training"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._experiment1 = MeshWriter(self.log_dir)

    @property
    def experiment1(self) -> MeshWriter:
        return self._experiment1
