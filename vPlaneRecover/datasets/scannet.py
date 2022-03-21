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

import csv
import json
import os

import numpy as np
import random
import time
import  shutil

def prepare_scannet_scene(scene, path, plane_path, path_meta, val_list, verbose=2 ):
    """Generates a json file for a scannet scene in our common format

    This wraps all the data about a scene into a single common format
    that is readable by our dataset class. This includes paths to all
    the images, depths, etc, as well as other metadata like camera
    intrinsics and pose. It also includes scene level information like
    a path to the mesh.

    Args:
        scene: name of the scene.
            examples: 'scans/scene0000_00'
                      'scans_test/scene0708_00'
        path: path to the original data from http://www.scan-net.org/
        plane_path: path to plane fitting result
        path_meta: path to where the generated data is saved.
            This can be the same as path, but it is recommended to
            keep them seperate so the original data is not accidentally
            modified. The generated data is saved into a mirror directory
            structure.

    Output:
        Creates the file path_meta/scene/info.json
        
    JSON format:
        {'dataset': 'scannet',
         'path': path,
         'scene': scene,
         'file_name_mesh_gt': '',               #pth
         'file_name_seg_indices': '',           #pth to segs.json
         'file_name_seg_groups': '',            #pth to aggregation.json
         'instances': None,                     # {instance id : label} load from aggregation.json
         'frames': [{'file_name_image': '',     #pth
                     'file_name_depth': '',     #pth
                     'file_name_instance': '',  #pth
                     'intrinsics': intrinsics,  #val
                     'pose': pose,              #val
                     }
                     ...
                   ]
         }

    """

    if verbose>0:
        print('preparing %s'%scene)

    folder, scene = scene.split('/')
    has_semantic_labels = (folder=='scans')

    data = {'dataset': 'scannet',
            'path': path,
            'plane_path': plane_path,
            'scene': scene,
            'file_name_mesh_gt': 
                os.path.join(path, folder, scene, scene+'_vh_clean_2.ply'), # some previous experiments areis down with _vh_clean_2.labels.ply
            'file_name_seg_indices': '',
            'file_name_seg_groups': '',
            'file_name_plane_mesh': '',
            'file_name_plane_param': '',
            'instances': None,
            'frames': []
           }

    # get instance id to category id
    # NOTE: the 3D geometry and semantic label will be evaluated on original mesh,
    # but plane instance label will be evaluated on the projected mesh
    if has_semantic_labels:
        data['file_name_seg_indices'] = os.path.join(path, folder, scene,  scene+'_vh_clean_2.0.010000.segs.json')
        data['file_name_seg_groups'] =  os.path.join(path, folder, scene,  scene+'.aggregation.json')
        data['file_name_plane_mesh'] = os.path.join(plane_path, 'filter_mesh_no_gap', scene + '_planes.ply')
        data['file_name_plane_param'] = os.path.join(plane_path, 'refined_pln_param', scene + '_planes.npy')

        label_mapping = load_scannet_label_mapping() # {label name : label id}

        # dict mapping instance id to label id
        with open(data['file_name_seg_groups'], 'r') as info_f:
            seg_groups = json.load(info_f)['segGroups']
        data['instances'] = {seg['id']+1:
                                label_mapping[seg['label']] for seg in seg_groups} # {scene instance id : scannet semantic id}

    # get camera intrinsics
    # we use color camera intrinsics and resize depth to match
    with open(os.path.join(path, folder, scene, '%s.txt' % scene)) as info_f:
        info = [line.rstrip().split(' = ') for line in info_f]
        info = {key:value for key, value in info}
        intrinsics = [
            [float(info['fx_color']), 0, float(info['mx_color'])],
            [0, float(info['fy_color']), float(info['my_color'])],
            [0, 0, 1]]


    # get per frame info
    dpth_folder =  'depth' # 'render_depth' if has_semantic_labels and (scene not in val_list) else 'depth' #depth' #
    frame_ids = os.listdir(os.path.join(path, folder, scene, dpth_folder))
    frame_ids = [int(os.path.splitext(frame)[0]) for frame in frame_ids]
    frame_ids =  sorted(frame_ids)


    for i, frame_id in enumerate(frame_ids):
        if verbose>1 and i%25==0:
            print('preparing %s frame %d/%d'%(scene, i, len(frame_ids)))

        pose = np.loadtxt(os.path.join(path, folder, scene, 'pose', '%d.txt' % frame_id))

        # skip frames with no valid pose
        if not np.all(np.isfinite(pose)):
            continue

        frame = {'file_name_image': 
                     os.path.join(path, folder, scene, 'color', '%d.jpg'%frame_id),
                 'file_name_depth': 
                     os.path.join(path, folder, scene, dpth_folder, '%d.png'%frame_id),
                 'file_name_instance': 
                     os.path.join(path, folder, scene, 'instance-filt',
                         '%d.png'%frame_id) if has_semantic_labels else '',
                 'intrinsics': intrinsics, #copy multiple times
                 'pose': pose.tolist(),
                }
        data['frames'].append(frame)

    os.makedirs(os.path.join(path_meta, folder, scene), exist_ok=True)
    json.dump(data, open(os.path.join(path_meta, folder, scene, 'info.json'), 'w'))


def prepare_scannet_splits(path, path_meta, valid_scenes, build_ours=False):
    """ Generates txt files for each dataset split

    Creates a folder 'data' in the current working directory.
    In that folder we create a split.txt file for each split containing
    the full path to the info.json for each scene in the split (one per line).
    We use the standard train/val/test splits from scannet.

    Args:
        path: Path to the original scannet data. This is used to get their
            standard splits.
        path_meta: Path to generated files.

    Example txt file:
        path_meta/scans/scene0000_00/info.json
        path_meta/scans/scene0000_01/info.json
        path_meta/scans/scene0001_00/info.json
        ...

    """
    # copy the train, test, value file to meta_file
    os.makedirs('meta_file', exist_ok=True)
    splits = [('scannet_train.txt', 'scans', 'scannetv2_train.txt'),
              ('scannet_val.txt', 'scans', 'scannetv2_val.txt'),
              ('scannet_test.txt', 'scans_test', 'scannetv2_test.txt'),
             ]
    if not os.path.isfile(os.path.join(path, 'scannetv2_train.txt')) or \
        not os.path.isfile(os.path.join(path, 'scannetv2_val.txt')) or \
            not os.path.isfile(os.path.join(path, 'scannetv2_test.txt')):
        if build_ours:
            scene_list = [x for x in os.listdir(path) if 'scene' in x and 'tar' not in x]
            all_scenes = list(set([x[:-3] for x in scene_list]))
            num_scene = len(all_scenes)
            # build our own train, val, test list, Fengting 10/26
            # the test scene will never been seen in the train/val set, and we take 00 result of it
            test_scenes = random.sample((all_scenes), int(0.1*num_scene))
            test_list = [x + '_00' for x in test_scenes]
            with open(path + '/scannetv2_test.txt', 'w') as f: #NOTE we did not use this split in our work
                for tst in test_list:
                    f.write(tst + '\n')

            # to make sure the val scene has no overlap
            val_list = random.sample([x for x in (scene_list) if x[:-3] not in test_scenes], int(0.1*num_scene))
            with open(path + '/scannetv2_val.txt', 'w') as f:
                for val in val_list:
                    f.write(val + '\n')

            train_list = [x for x in scene_list if x[:-3] not in test_scenes and x not in val_list]
            with open(path + '/scannetv2_train.txt', 'w') as f:
                for train in train_list:
                    f.write(train + '\n')
            time.sleep(1) # pause 1 sec to ensure writing
        else:
            print('please download the train/val/test split from https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark\t\n '
                  'and place them under --path and try again')
            exit(1)

    for name, folder, fname in splits:
        with open(os.path.join('meta_file', name), 'w') as out_file:
            scenes = [line.rstrip() for line in sorted(open(os.path.join(path, fname), 'r'))]
            for scene in scenes:
                if 'train' in name  and  ('scans/{}'.format(scene) not in valid_scenes):
                    continue

                if 'val' in name and scene[-2:] != '00':
                    continue

                json_file = os.path.join(path_meta, folder, scene, 'info.json') #
                out_file.write(json_file+'\n')
                print(json_file)

    if not os.path.isfile('meta_file/scannetv2-labels.combined.tsv'):
        # copy mapping file into dataset splits for easy access
        shutil.copy('%s/scannetv2-labels.combined.tsv'%path, 'meta_file/scannetv2-labels.combined.tsv')



def load_scannet_label_mapping(pth=None):
    """ Returns a dict mapping scannet category label strings to scannet Ids

    scene****_**.aggregation.json contains the category labels as strings 
    so this maps the strings to the integer scannet Id

    Args:
        path: Path to the original scannet data.
              This is used to get scannetv2-labels.combined.tsv

    Returns:
        mapping: A dict from strings to ints
            example:
                {'wall': 1,
                 'chair: 2,
                 'books': 22}

    """

    mapping = {}
    if pth is None:
        pth = os.path.join('meta_file', 'scannetv2-labels.combined.tsv')
    with open(pth) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for i, line in enumerate(tsvreader):
            if i==0:
                continue
            id, name = int(line[0]), line[1]
            mapping[name] = id

    return mapping


def load_scannet_nyu40_mapping(pth=None):
    """ Returns a dict mapping scannet Ids to NYU40 Ids

    Args:
        path: Path to the original scannet data. 
            This is used to get scannetv2-labels.combined.tsv

    Returns:
        mapping: A dict from ints to ints
            example:
                {1: 1,
                 2: 5,
                 22: 23}

    """

    mapping = {}
    if pth is None:
        pth = os.path.join('meta_file', 'scannetv2-labels.combined.tsv')
    with open(pth) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for i, line in enumerate(tsvreader):
            if i==0:
                continue
            id, nyu40id = int(line[0]), int(line[4])
            mapping[id] = nyu40id
    return mapping
