import numpy as np
import trimesh

class ColorPalette:
    def __init__(self, numColors):
        np.random.seed(2)
        self.colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [255, 230, 180],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],
                                  [255, 255, 0],
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],
                                  [0, 100, 100],
                                  [0, 255, 128],
                                  [0, 128, 255],
                                  [255, 0, 128],
                                  [128, 0, 255],
                                  [255, 128, 0],
                                  [128, 255, 0],
        ])

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.concatenate([self.colorMap, np.random.randint(255, size = (numColors - self.colorMap.shape[0], 3))], axis=0)
            pass

        return

    def getColorMap(self):
        return self.colorMap

    def getColor(self, index):
        if index >= self.colorMap.shape[0]:
            return np.random.randint(255, size = (3))
        else:
            return self.colorMap[index]


def writePointCloudFace(filename, points, faces):
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0
element vertex """
        header += str(len(points))
        header += """
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face """
        header += str(len(faces))
        header += """
property list uchar int vertex_index
end_header
"""
        f.write(header)
        for point in points:
            for value in point[:3]:
                f.write(str(value) + ' ')
                continue
            for value in point[3:]:
                f.write(str(int(value)) + ' ')
                continue
            f.write('\n')
            continue
        for face in faces:
            f.write('3 ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + '\n')
            continue
        f.close()
        pass
    return


def load_plane_map():
    # none plane class : pillow  toilet bag curtain lamp shower curtain person clothes mirror
    nyu_labels = ['bed', 'books', 'toilet', 'chair', 'sofa', 'pillow', 'desk', 'bathtub', 'bag', 'wall',
                  'picture', 'floor', 'curtain', 'lamp', 'shower curtain', 'floor mat', 'sink', 'window', 'otherprop',
                  'counter', 'box', 'person', 'ceiling', 'otherstructure',  'refridgerator', 'bookshelf', 'shelves', 'door',
                  'otherfurniture', 'television', 'mirror',  'towel', 'dresser', 'whiteboard', 'cabinet',
                  'clothes', 'paper',  'blinds',  'table', 'night stand','unannotated']
    plane_map = dict()
    confident_categories = ['wall','picture','cabinet','whiteboard','floor','bed','door','chair',
                            'sofa','table','counter','shelves','ceiling']

    neutral_categories = ['blinds','books','sink','paper','window','refridgerator','television',
                          'bathtub','otherprop','box','otherstructure','otherfurniture','night stand','unannotated']
    for label in nyu_labels:
        if label in confident_categories:
            plane_map[label] = label
        elif label == 'floor mat':
            plane_map[label] = 'floor'
        elif label == 'bookshelf' :
            plane_map[label] = 'shelves'
        # elif label == 'blackboard':
        #     plane_map[label] = 'whiteboard'
        elif label == 'desk':
            plane_map[label] = 'table'
        elif label == 'dresser':
            plane_map[label] = 'counter'
        elif label in neutral_categories:
            plane_map[label] = 'other_plane'
        else:
            plane_map[label] = 'non_plane'

    return plane_map

def load_num_map():
    # OldlabelNumPlanes = {
    #         'wall':[1,3],
    #         'picture':[1,1],
    #         'floor':[1,1],
    #         'board':[1,5],
    #         'bed':[1,5],
    #         'door':[1,2],
    #         'chair':[1,2],
    #         'sofa':[1,10],
    #         'table':[1,10],
    #         'counter':[1,10],
    #         'cabinet':[1,10],
    #         'shelves':[1,5],
    #         'ceiling':[1,5],
    #         'other_plane':[0,5],
    #         'non_plane':[0,0]
    #         }

    labelNumPlanes = {
            'wall':[1,3],
            'picture':[1,1],
            'floor':[1,1],
            'whiteboard':[1,5],
            'bed':[1,5],
            'door':[1,2],
            'chair':[0,10],
            'sofa':[1,10],
            'table':[1,5],
            'counter':[1,5],
            'cabinet':[1,5],
            'shelves':[1,5],
            'ceiling':[1,5],
            'other_plane':[0,5],
            'non_plane':[0,0]
            }

    return labelNumPlanes

def check_nparray(a, b):
    set_a, set_b = set(), set()
    for x in a.tolist():
        set_a.add(tuple(x))
    for y in b.tolist():
        set_b.add(tuple(y))

    print(len(set_a.difference(set_b)))


def fitPlane(points, norm_in=None, norm_thres=0):
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        if norm_in is None:
            return np.linalg.lstsq(points, np.ones(points.shape[0]), rcond=None)[0]
        else:
            norm = norm_in.copy()
            # to control the memory usage, svd take lots of memory if the vector is too long
            if norm_in.shape[0] > 10000:
                mask =  np.random.choice(np.arange(norm.shape[0]), size=(10000), replace=False)
                norm = norm[mask]

            u,s,v = np.linalg.svd((norm))
            planeN = v[0, :] # here numpy do decreasing order and transpose
            # because scannet use normal to do over-segmt, here each points has similair normal, no need to further filtering
            mean_pnt = np.mean(points, 0, keepdims=True)
            planeD = mean_pnt.dot(planeN)
            return planeN / planeD



def planeRCNN_ransac(XYZ_in, sgmt_pntsId_in, norm_in=None, n_iter=100, n_plane=2, diff_thres=0.05, area_thres=10, norm_thres=np.cos(np.deg2rad(30))):
    XYZ = XYZ_in.copy()
    sgmt_pntsId = sgmt_pntsId_in.copy()
    norm = norm_in.copy()

    segmentPlanes = []
    segmentPlanePointIndices = []

    for planeIndex in range(n_plane):
        if len(XYZ) < area_thres:
            continue
        bestPlaneInfo = [None, 0, None]
        for iteration in range(min(XYZ.shape[0], n_iter)):
            sampledPoints = XYZ[np.random.choice(np.arange(XYZ.shape[0]), size=(3), replace=False)]
            try:
                plane = fitPlane(sampledPoints, norm, norm_thres=norm_thres)
            except:
                continue
            diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)
            inlierMask = diff < diff_thres
            numInliers = inlierMask.sum()
            if numInliers > bestPlaneInfo[1]:
                bestPlaneInfo = [plane, numInliers, inlierMask]

        if bestPlaneInfo[1] < area_thres:
            print("Too small plane, ignore")
            break

        pointIndices = sgmt_pntsId[bestPlaneInfo[2]]
        bestPlane = fitPlane(XYZ[bestPlaneInfo[2]])

        segmentPlanes.append(bestPlane)
        segmentPlanePointIndices.append(pointIndices)

        # the remain points to fit the next plane
        outlierMask = np.logical_not(bestPlaneInfo[2])
        sgmt_pntsId = sgmt_pntsId[outlierMask]
        XYZ = XYZ[outlierMask]
        norm = norm[outlierMask]

    return segmentPlanes, segmentPlanePointIndices, sgmt_pntsId

def onePoint_ransac(XYZ_in, sgmt_pntsId_in, norm_in, n_iter=100, n_plane=2, diff_thres=0.05, area_thres=10, norm_thres=0.9):
    XYZ = XYZ_in.copy()
    norm = norm_in.copy()
    sgmt_pntsId = sgmt_pntsId_in.copy()

    segmentPlanes = []
    segmentPlanePointIndices = []

    for planeIndex in range(n_plane):
        if len(XYZ) < area_thres:
            continue
        bestPlaneInfo = [None, 0, None]

        iter = min(XYZ.shape[0], n_iter)
        rand_IDs = np.random.choice(np.arange(XYZ.shape[0]), size=(iter), replace=False)

        for iteration in range(iter):
            rand_ID = rand_IDs[iteration]
            sampledPnt = XYZ[rand_ID]
            sampledNorm = norm[rand_ID]

            plane = sampledNorm / sampledNorm.dot(sampledPnt.T)
            if np.linalg.norm(plane) < 1e-4:
                continue

            left_eq = XYZ.dot(plane.T)
            diff = np.abs(left_eq - np.ones_like(left_eq)) / np.linalg.norm(plane)

            dist_inlierMask = diff < diff_thres
            norm_inlierMask = (np.abs(norm).dot(np.abs(sampledNorm).T))  > norm_thres

            inlierMask = np.logical_and(dist_inlierMask, norm_inlierMask) #dist_inlierMask #

            numInliers = inlierMask.sum()
            if numInliers > bestPlaneInfo[1]:
                bestPlaneInfo = [plane, numInliers, inlierMask.reshape(-1)]

        if bestPlaneInfo[1] < area_thres:
            print("Too small plane, ignore")
            break

        pointIndices = sgmt_pntsId[bestPlaneInfo[2]]
        bestPlane = fitPlane(XYZ[bestPlaneInfo[2]], norm[bestPlaneInfo[2]])

        # print((bestPlane), bestPlaneInfo[1], bestPlaneInfo[2].sum())

        segmentPlanes.append(bestPlane)
        segmentPlanePointIndices.append(pointIndices)

        # the remain points to fit the next plane
        outlierMask = np.logical_not(bestPlaneInfo[2])
        sgmt_pntsId = sgmt_pntsId[outlierMask]
        XYZ = XYZ[outlierMask]
        norm = norm[outlierMask]

    return segmentPlanes, segmentPlanePointIndices, sgmt_pntsId

def vis_meshNormal(mesh, normal):
    _mesh = mesh.copy()
    _mesh.visual.vertex_colors = normal
    _mesh.show()

def get_nyu_id2labl():
    ids = list(range(0, 41))
    labels = ['unannotated', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                        'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat',
                         'clothes', 'ceiling', 'books', 'refrigerator', 'television', 'paper', 'towel', 'shower curtain',
                        'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag',
                        'otherstructure', 'otherfurniture', 'otherprop']
    assert len(ids) == len(labels)
    id2label = {}
    for id, label in zip(ids,labels):
       id2label[id] = label

    return id2label

import os
import csv

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


NYU40_COLORMAP = [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),
       (247, 182, 210),		# desk
       (66, 188, 102),
       (219, 219, 141),		# curtain
       (140, 57, 197),
       (202, 185, 52),
       (51, 176, 203),
       (200, 54, 131),
       (92, 193, 61),
       (78, 71, 183),
       (172, 114, 82),
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),
       (153, 98, 156),
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),
       (227, 119, 194),		# bathtub
       (213, 92, 176),
       (94, 106, 211),
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]