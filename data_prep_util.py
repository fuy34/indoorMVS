import numpy as np
import torch
import torch.nn.functional as F

def normal_angle(norm_in):
    # note: norm(norm) == 1
    # we only consider up semi-sphere
    norm = norm_in.copy()
    # norm[norm[:, 2] <0] *= -1

    phi = np.rad2deg(np.arccos(norm[:, 2]))  # pitch
    # phi = 0 if phi == 180 else phi # 0 and 180 are same cls , there is no perfect 180 or 0 in the fitting data

    xy = np.sqrt(norm[:, 0] ** 2 + norm[:, 1] ** 2)
    theta = np.rad2deg(np.arccos(norm[:, 0] / xy))  # yaw, np.arccos only return 0, pi
    theta[norm[:, 1] < 0] = 360 - theta[norm[:, 1] < 0] # build 360 deg result
    # theta = 0 if theta == 180 else theta

    return theta, phi

def map_planes(vert, vert_colors,  plane_param,  angle_interval=30):
    # c2v_dire: centroid to vertices direction vec
    plane_verts = vert.view(np.ndarray).copy()
    if  180 % angle_interval != 0 and 90 % angle_interval !=0 or 360 % angle_interval != 0:
        print('anlge_interval must be dividable to 90, 180, 360')
        exit(1)

    # convert plane param
    # note we save plane_param as n*d
    planeD = np.sqrt(np.sum(plane_param * plane_param, axis=1, keepdims=True)) # d
    planeN = plane_param / planeD

    # convert color to plane id
    invalid_id = 16777216 // 100 -1 #(255,255,255) refer to fit_plane code get_gtPlane_segmt.py
    chan_0 = vert_colors[:, 2]
    chan_1 = vert_colors[:, 1]
    chan_2 = vert_colors[:, 0]
    plane_id = (chan_2 * 256 ** 2 + chan_1 * 256 + chan_0) // 100 - 1  # there is no (0,0,0) color in fitting mesh

    # assign plane ins, we get plane center in voxel space to ensure each center has same ins id and normal as instance
    unique_id = np.unique(plane_id) # will return sorted unique elements of an array
    plane_ins = np.zeros_like(plane_id, dtype=np.uint32)

    # ins == 1 means correspond to the first plane centers, use 0 as non plane
    plane_id2param = np.zeros([unique_id.shape[0], 3])
    for k, id in enumerate(unique_id):
        if id == invalid_id: continue
        plane_ins[plane_id == id] = (k + 1)  #
        plane_id2param[k+1] = plane_param[id] / planeD[id] /planeD[id]

        # proj plane idx towards the plane
        cluster_plane_dist = (planeN[id:id+1] @ vert[plane_id == id].T) - planeD[id]
        dist_sign = np.sign(cluster_plane_dist)
        plane_verts[plane_id==id] = vert[plane_id==id] - (dist_sign.T * (np.abs(cluster_plane_dist).T @ planeN[id:id+1]))



    return  plane_ins, plane_id2param, plane_verts #plane_normal, plane_cls,


def get_normal( tsdf_vol):
    # refer to https://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
    # Note the tsdf coordiate are x y z
    # mask = ~torch.logical_or (tsdf_vol == 1, tsdf_vol==-1)
    # replicate usage
    if len(tsdf_vol.shape) == 3:
        tsdf_vol = tsdf_vol.unsqueeze(0).unsqueeze(0)
    pad_vol = F.pad(tsdf_vol, (1, 1, 1, 1, 1, 1),
                    mode="replicate")  # pad each dim 1,1 to compute grad
    nx = (pad_vol[:,:, 2:, :, :] - pad_vol[:,:, :-2, :, :])[:,:, :, 1:-1, 1:-1]
    ny = (pad_vol[:,:, :, 2:, :] - pad_vol[:,:, :, :-2, :])[:,:, 1:-1, :, 1:-1]
    nz = (pad_vol[:,:, :, :, 2:] - pad_vol[:,:, :, :, :-2])[:,:, 1:-1, 1:-1, :]

    normal = torch.cat([nx, ny, nz], dim=1) # concat in channel dim

    normal /= normal.norm(dim=1)
    normal[normal != normal] = 0  # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4 set nan to 0
    return normal.squeeze()
