import numpy as np
import torch
import sys
from collections import deque
# prob_thres = 0.5
# # spatial_thres = 3
# norm_thres = np.cos(np.deg2rad(30))
# RADIUS =

import torch.nn.functional as F
# from vPlaneRecover.util import coordinates
import trimesh
from skimage import measure

# grouping the 3 sigma region defined in prepare_data
GROUPING_RADIUS =  {16: 1, 8: 2, 4: 3}

#  window, blind, pillow, mirror, clothes, shower curtain, person, toilet, sink, lamp, bag
NONE_PLANE_ID = [0, -1, 9, 13, 18, 19, 21, 28, 31, 33, 34,  35, 37 ]

# for generating
STD =  {16: 1 ,8: 2, 4: 3 }
ADOPT_THRES = 0.03
PI = torch.tensor(3.1415926)
SQRT_2PI = 2.506628

ORTHOGONAL_THRES = np.cos(np.deg2rad(60))
# for inference
PLANE_MIN_N_VOXELS = {16: 25 ,8: 100, 4: 400 } #0.04**3*1600=0.1024 m3 -- marchine cube need at least 2

# SEM_INS_MAP = {1:0, 2:1, 22:2, 30:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 10:10,
#                11:11, 12:12, 14:13, 16:14, 24:15, 28:16, 33:17, 36:18, 39:19}
#
# CLASS_LABELS = ['wall', 'floor', 'ceiling', 'whiteboard', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
#                 'bookshelf', 'picture',  'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'bathtub', #'sink', 'window',
#                 'otherfurniture']

# SEM_INS_MAP = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9,
#                11:10, 12:11, 14:12, 16:13, 24:14,   33:15, 36:16, 39:17, 22:18, 30:19}
#
# CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
#                 'bookshelf', 'picture',  'counter', 'desk', 'curtain', 'refrigerator',   'toilet', 'bathtub', #'sink', 'window',
#                 'otherfurniture', 'ceiling', 'whiteboard']

SEM_INS_MAP = {1:0, 2:1, 22:2} # 8:2,

CLASS_LABELS = ['wall', 'floor','ceiling'] #'door',

LAYOUT_SEM = {1, 2,22, 30, 3, 7, 12, 14, 36}
# ============= winner take all assignment (deprecated) ===========================
def get_centers(voxel_coord, voxel_norm, semsegs, center_prob_in, surface_mask, center_thres, radius, norm_thres, topk=5000):
    # center_prob = center_prob.clamp(0, 1)

    # pick top k, to save memory
    center_prob = center_prob_in.reshape([-1])
    _, indices = torch.sort(center_prob[surface_mask.reshape([-1])], descending=True)
    ori_indx = surface_mask.reshape([-1]).nonzero(as_tuple=False)
    _idx = indices[:topk]
    idx = ori_indx[_idx]
    valid_mask = torch.zeros_like(center_prob).type(torch.bool)
    valid_mask[idx] = (center_prob[idx] > center_thres)
    valid_mask = valid_mask.view(center_prob_in.size())

    if valid_mask.sum() == 0:
        return [], [], []

    # get all inliners
    inlier_coord, inlier_norm= voxel_coord[:, valid_mask.reshape([-1])], voxel_norm[:, valid_mask].reshape([3,-1])
    inlier_semseg, inlier_prob = semsegs[valid_mask].reshape([-1]), center_prob[valid_mask.reshape([-1])]

    # sort prob
    desc_prob, indices = torch.sort(inlier_prob, descending=True)
    tmp_indices = indices.clone()
    norm_ignore_mask =  (torch.sum(inlier_norm[:, indices], dim=0) == 0)
    sem_ignore_mask = (inlier_semseg[indices] == 0)
    centers = []
    center_segm_lst = []
    center_norm_lst = []
    idx = 0
    while idx < tmp_indices.shape[0]:
        if tmp_indices[idx] != -1:
            cur_center = inlier_coord[:, tmp_indices[idx]].reshape([3,1])
            center_norm = inlier_norm[:, tmp_indices[idx]].reshape([3,1])
            center_semseg = inlier_semseg[tmp_indices[idx]].reshape([1])


            # group the surrounding ones if they share same semantic label and the normal dist. < cos(30),
            sp_dist = torch.sum((inlier_coord[:, indices] - cur_center) ** 2, dim=0)

            # if no valid norm, we ingore the norm dist constrain
            if (torch.sum(center_norm.abs(), dim=0) == 0):
                norm_mask = torch.ones_like(center_semseg).type(torch.bool)
            else:
                norm_mask = (torch.sum((inlier_norm[:, indices] * center_norm), dim=0).abs()> norm_thres) | norm_ignore_mask

            # if no valid semseg, we ingore the semseg dist
            if center_semseg == 0:
                semseg_mask = torch.ones_like(center_semseg).type(torch.bool)
            else:
                semseg_mask = ((center_semseg == inlier_semseg[indices])  | sem_ignore_mask)

            center_mask = (sp_dist < radius * radius) & semseg_mask  & norm_mask
            tmp_indices[center_mask] = -1      # nms

            # set the center as the weighted sum of the highest one's surrounding --- this will cause center fell outside
            # of the semantic area, just pick the highest on
            # new_center =((inlier_coord[:, indices[center_mask]] * inlier_prob[indices[center_mask]]).sum(dim=-1) / \
            #              (inlier_prob[indices[center_mask]]).sum(dim=-1)).round().type(torch.int)

            # find the nearest voxel has semantic and normal
            # new_center_norm = voxel_norm[:, new_center[0], new_center[1], new_center[2]]
            # new_center_semg = semsegs[new_center[0], new_center[1], new_center[2]]
            # dist = torch.sum((inlier_coord[:, indices] - new_center.unsqueeze(1)) ** 2, dim=0)
            _, new_idx = torch.sort(sp_dist) #acedend
            new_center = cur_center.clone()
            new_center_norm = center_norm.clone()
            new_center_semg = center_semseg.clone()
            cnt = 0
            while ((torch.sum(new_center_norm.abs(), dim=0) == 0) or (new_center_semg == 0)) and (cnt < new_idx.shape[0]):
                cur_idx = new_idx[cnt]
                new_center = inlier_coord[:, indices][:, cur_idx].type(torch.long).view(cur_center.shape)
                new_center_norm = voxel_norm[:, new_center[0], new_center[1], new_center[2]].view(center_norm.shape)
                new_center_semg = semsegs[new_center[0], new_center[1], new_center[2]].view(center_semseg.shape)
                cnt+=1

            # if the whole ball is in empty space
            if ((torch.sum(new_center_norm.abs(), dim=0) == 0) or (new_center_semg == 0)):
                continue
            else:
                centers.append(new_center)
                center_segm_lst.append(new_center_semg)
                center_norm_lst.append(new_center_norm)

            # debug
            # if idx == 0:
            #     vert_viz = np.concatenate([vert, inlier_coord.T.numpy()], axis=0)
            #     vert_color, voxel_color = np.zeros([vert.shape[0], 3]), np.ones([inlier_coord.shape[1], 3])*255
            #     voxel_color[:,2] = 0
            # voxel_color[indices.numpy()[center_mask.numpy()]] =  np.array([0,0,255])
            # print(center_mask.sum(), new_center.round().type(torch.int))
            # color_viz = np.concatenate([vert_color, voxel_color], axis=0).astype(np.int)
            # pld = trimesh.points.PointCloud(vertices=vert_viz,  colors=color_viz, process=False)
            # pld.show()

        idx += 1

    return centers, center_segm_lst, center_norm_lst

def get_planeIns(tsdf, cfg):

    voxel_size =  tsdf.voxel_size
    normals =   tsdf.attribute_vols['plane_norm']
    semIns = tsdf.attribute_vols['semseg']
    center_prob = tsdf.attribute_vols['centroid_prob']

    mask_surface = tsdf.tsdf_vol.abs() < 1


    radius =  GROUPING_RADIUS[int(voxel_size * 100)]
    norm_thres =  np.cos(np.deg2rad(cfg.MODEL.GROUPING.NORM_THRES))
    prob_thres =  cfg.MODEL.GROUPING.PROB_THRES
    topk_prob = cfg.MODEL.GROUPING.TOPK_PROB

    coords = coordinates(center_prob.shape,device=tsdf.device)
    centers, center_segms, center_norms = get_centers( coords, normals, semIns, center_prob, mask_surface,
                                                     prob_thres, radius,norm_thres, topk_prob)

    normals, semIns = normals.reshape([3, -1]), semIns.reshape([-1])
    planeIns = torch.zeros_like(semIns)

    for i, (center_coord, center_seg, center_norm) in enumerate(zip(centers, center_segms, center_norms)):
        # semantic should be same
        semseg_mask = (semIns == center_seg)

        # normal should be similiar
        norm_mask = (torch.sum((normals * center_norm), dim=0).abs() > norm_thres)

        # distance to the plane should under threshold
        planeD = (center_norm * center_coord).sum()
        cluster_plane_dist = ((center_norm * coords).sum(dim=0) - planeD).abs()
        spatial_mask = cluster_plane_dist <= radius

        # only assign once with highest prob center
        # note we do not prevent a center to be assigned to another center with higher prob, so planeIns.max()
        # is not equal to the plane_num
        available_mask = planeIns == 0

        # assign plane instance to the picked voxels
        cluster_mask = semseg_mask & norm_mask & spatial_mask & available_mask

        # some center may do not have ins because of the limited number of inliner
        if cluster_mask.sum() > PLANE_MIN_N_VOXELS[int(voxel_size * 100)]:
            planeIns[cluster_mask] = (i + 1)
            normals[:, cluster_mask] = center_norm
        # print(cluster_mask.sum())

    return planeIns, normals,  centers, center_segms, center_norms

# ============= RANSAC based assignment =====================
def check_connection_bfs(mask, seed_idx, vol_shape):
    # bfs to build connection mask --- deprecated too slow !!!
    # vol_shape: h, w, d, where h w in 2D, d is the height dim
    h, w, d = vol_shape
    vol_mask = mask.view(vol_shape)

    # set the seed
    candidate_mask = torch.zeros_like(mask).bool()
    candidate_mask[seed_idx] = True
    candidate_mask = candidate_mask.view(vol_shape)
    idx = (candidate_mask == True).nonzero(as_tuple=False).squeeze().cpu().numpy() #shape (1,3), x y z

    # 6 direction search
    x, y, z = idx[0], idx[1], idx[2]
    queue = deque([(x, y, z)])
    while len(queue) > 0:
        cur_x, cur_y, cur_z = queue.popleft()
        sys.stdout.write('\r connection: {}/{}'.format(candidate_mask.sum(), mask.sum()))
        sys.stdout.flush()

        if cur_x - 1 >= 0 and vol_mask[cur_x-1, cur_y, cur_z] == True and candidate_mask[cur_x-1, cur_y, cur_z] == False:
            candidate_mask[cur_x-1, cur_y, cur_z] = True
            queue.append((cur_x-1, cur_y, cur_z))

        if cur_x + 1 < h and vol_mask[cur_x +1, cur_y, cur_z] == True and candidate_mask[cur_x+1, cur_y, cur_z] == False:
            candidate_mask[cur_x+1, cur_y, cur_z] = True
            queue.append((cur_x +1, cur_y, cur_z))

        if cur_y - 1 >= 0 and vol_mask[cur_x, cur_y - 1, cur_z] == True and candidate_mask[cur_x, cur_y-1, cur_z] == False:
            candidate_mask[cur_x , cur_y -1, cur_z] = True
            queue.append((cur_x, cur_y - 1, cur_z))

        if cur_y + 1 < h and vol_mask[cur_x, cur_y + 1, cur_z] == True and candidate_mask[cur_x, cur_y+1, cur_z] == False:
            candidate_mask[cur_x, cur_y + 1, cur_z] = True
            queue.append((cur_x, cur_y + 1, cur_z))

        if cur_z - 1 >= 0 and vol_mask[cur_x, cur_y, cur_z - 1] == True and candidate_mask[cur_x, cur_y, cur_z-1] == False:
            candidate_mask[cur_x, cur_y, cur_z - 1] = True
            queue.append((cur_x, cur_y, cur_z - 1))

        if cur_z + 1 < h and vol_mask[cur_x, cur_y, cur_z + 1] == True and candidate_mask[cur_x, cur_y, cur_z+1] == False:
            candidate_mask[cur_x , cur_y, cur_z + 1] = True
            queue.append((cur_x, cur_y, cur_z + 1))

    return candidate_mask.view(-1)

def check_connection(mask, seed_idx, vol_shape):
    # use make pooling to keep update until it reach
    vol_mask = mask.view(vol_shape).unsqueeze(0).unsqueeze(0)

    # set the seed
    candidate_mask = torch.zeros_like(mask).bool()
    candidate_mask[seed_idx] = True
    candidate_mask = candidate_mask.view(vol_shape).unsqueeze(0).unsqueeze(0)

    memo_mat = candidate_mask.clone()
    pre_mask = candidate_mask.clone()

    # use 3D max pooling to propgate seed
    for cnt in range(max(vol_shape)): # longest dist to flood fill equals to the largest dim
        candidate_mask = F.max_pool3d(candidate_mask.float(), kernel_size=3, stride=1, padding=1).bool() & vol_mask
        memo_mat = F.max_pool3d(memo_mat.float(), kernel_size=3, stride=1, padding=1)
        if memo_mat.sum() >= vol_shape[0] * vol_shape[1] * vol_shape[2] or (candidate_mask == vol_mask).all()\
                or (pre_mask == candidate_mask).all():
            break
        pre_mask = candidate_mask.clone()
        # sys.stdout.write('\rflood fill step: {}'.format(cnt))
        # sys.stdout.flush()
    return candidate_mask.view(-1)


def seq_ransac(coords, normals, semLab, prob, planeIns, valid_mask, mask_surface,
               norm_thres, radius, area_thres, cur_planeID,  vol_shape, n_iter=100):
    # sequential one-point plane ransac, the principle is the same as onePoint_ransan in fit_plane/util,
    # but the code is slightly different  because we do not have the instance level label,
    # so we need consider semantic label consistency here

    # sample the seeds w.r.t. their prob
    prob[~valid_mask] = 0

    # in case replacement == False, and n_iter > (prob > 0).sum(), it will return idx whose weight == 0
    # therefore, we should make sure the n_iter is always < (prob>0).sum()
    idxs = torch.multinomial(prob, min(n_iter, (prob > 0).sum()), replacement=False)

    resume = False
    n_inliers = 0
    cnt = 0
    best_mask = torch.zeros_like(semLab).type(torch.bool)
    for i in idxs:
        cnt += 1
        sample_pnt = coords[:, i].unsqueeze(1)
        sample_norm = normals[:, i].unsqueeze(1)
        sample_semg = semLab[i]

        sys.stdout.write('\rprocessing: {}, assigning plane: {}, iter {}'.format(sample_semg.item(), cur_planeID, cnt))
        sys.stdout.flush()

        # semantic should be same
        semseg_mask = (semLab == sample_semg)

        # normal should be similiar
        norm_mask = (torch.sum((normals * sample_norm), dim=0).abs() > norm_thres)

        # distance to the plane should under threshold
        planeD = (sample_norm * sample_pnt).sum()
        cluster_plane_dist = ((sample_norm * coords).sum(dim=0) - planeD).abs()
        spatial_mask = cluster_plane_dist <= radius

        # only sign once
        available_mask = planeIns == 0

        # connection mask -- only if the voxel have path connected to the sample pnt can be assigned
        cluster_mask = semseg_mask & norm_mask & spatial_mask & available_mask & mask_surface


        # put all the masks together --- Ideally we should put connection mask here, but it is too slow, we do it in post process instead
        # cluster_mask = cluster_mask & connection_mask

        n =  cluster_mask.sum()
        if n > n_inliers:
            best_mask = cluster_mask.clone()
            n_inliers = n
            center_idx, center_coord, center_semseg, center_norm = i, sample_pnt, sample_semg, sample_norm

    # best_mask = best_mask | (planeIns == cur_planeID) # for NN case
    # ransac will stop if the best plane_area < area_thres
    if n_inliers >= area_thres:
        # break the mask if contain 2 seperate part
        connection_mask = check_connection(best_mask, center_idx, vol_shape)
        best_mask = connection_mask & best_mask

        # only update label if the area is sufficent large
        if best_mask.sum() < area_thres:
            return planeIns, valid_mask, resume, normals, sample_pnt, sample_semg, sample_norm
        # ======= debug========
        # voxels = coords[:, mask_surface].T.cpu().numpy()
        # voxel_color = np.zeros([voxels.shape[0], 3])
        # voxel_color[best_mask.cpu().numpy()[mask_surface.cpu().numpy()]] =  np.array([0,0,255])
        #
        # pld = trimesh.points.PointCloud(vertices=voxels, colors=voxel_color, process=False)
        # pld.show()
        #
        # best_mask = connection_mask & best_mask
        # voxel_color[best_mask[mask_surface].cpu().numpy()] = np.array([255, 0, 0])
        # pld = trimesh.points.PointCloud(vertices=voxels, colors=voxel_color, process=False)
        # pld.show()

        # -------- connect
        # if sample_semg ==3:
        #     pld = trimesh.points.PointCloud(vertices=voxels, colors=voxel_color, process=False)
        #     pld.show()

            # vol_mask = best_mask.view(vol_shape).unsqueeze(0).unsqueeze(0)
            #
            # # set the seed
            # candidate_mask = torch.zeros_like(best_mask).bool()
            # candidate_mask[center_idx] = True
            # candidate_mask = candidate_mask.view(vol_shape).unsqueeze(0).unsqueeze(0)
            #
            # memo_mat = candidate_mask.clone()
            # pre_mask = candidate_mask.clone()
            # # use 3D max pooling to propgate seed
            # for cnt in range(max(vol_shape)):  # longest dist to flood fill equals to the largest dim
            #     candidate_mask = F.max_pool3d(candidate_mask.float(), kernel_size=3, stride=1, padding=1).bool() & vol_mask
            #
            #     connection_mask = candidate_mask.view(-1)
            #     voxel_color[connection_mask[mask_surface].cpu().numpy()] = np.array([255, 0, 0])
            #     pld = trimesh.points.PointCloud(vertices=voxels, colors=voxel_color, process=False)
            #     pld.show()
            #
            #     memo_mat = F.max_pool3d(memo_mat.float(), kernel_size=3, stride=1, padding=1)
            #     if memo_mat.sum() >= vol_shape[0] * vol_shape[1] * vol_shape[2] or (candidate_mask == vol_mask).all():
            #         break

        planeIns[best_mask] = cur_planeID
        valid_mask[best_mask] = False # seed will only be assigned once as well
        resume = True

        # get weird result -- perform least square fit in the ininlier -- we should not change normal, because the voxel norm if from tsdf_grad
        # it can be very different in the boundary

        mean_center = torch.mean(coords[:,best_mask].float(), dim=1, keepdim=True)
        dist = torch.sum((coords[:, best_mask].float() - mean_center).abs(), dim=0) # use l1 dist find nearest inliers to mean_center
        new_idx = dist.argmin(dim=0)  # acedend
        center_coord = coords[:,best_mask][:, new_idx].unsqueeze(1)
        normals[:, best_mask] = center_norm
    else:
        center_coord, center_semseg, center_norm = sample_pnt, sample_semg, sample_norm # just return sth, will not be used

    return planeIns, valid_mask, resume, normals,  center_coord, center_semseg, center_norm

def get_planeIns_RANSAC(tsdf, cfg):
    # init necessary variables
    voxel_size =  tsdf.voxel_size
    normals =   tsdf.attribute_vols['plane_norm']
    semLab = tsdf.attribute_vols['semseg']
    center_prob = tsdf.attribute_vols['centroid_prob']

    radius =  GROUPING_RADIUS[int(voxel_size * 100)]
    norm_thres =  np.cos(np.deg2rad(cfg.MODEL.GROUPING.NORM_THRES))
    prob_thres =  cfg.MODEL.GROUPING.PROB_THRES
    area_thres =  PLANE_MIN_N_VOXELS[int(voxel_size * 100)]

    coords = coordinates(center_prob.shape,device=tsdf.device)

    normals, semLab = normals.reshape([3, -1]), semLab.reshape([-1])
    planeIns = torch.zeros_like(semLab)

    # pick valid center
    mask_surface = tsdf.tsdf_vol.abs() < 1
    center_prob_flat = center_prob.clone().reshape([-1])
    seed_mask = (center_prob_flat > prob_thres) & mask_surface.reshape([-1])

    # _, indices = torch.sort(center_prob_flat[seed_mask], descending=True)
    # ori_indx = seed_mask.nonzero(as_tuple=False)
    # idx_in_ori_idx = ori_indx[indices] # convert the valid set idx to the whole set idx

    if seed_mask.sum() == 0:
        return planeIns, normals, [], [], []

    # start sequential RANSAC for each pred_semantic label
    cur_planeId = 1
    centers, center_segms, center_norms = [], [], []
    for semid in torch.unique(semLab):
        if semid <= 0: continue # ignore invalid semantic label
        resume_ransac = True
        tmp_valid_mask = seed_mask & (semLab == semid)
        # if semid == 22:
        #     print(123)
        # Start seq_Ransac,
        while resume_ransac:
            if tmp_valid_mask.sum() == 0: #quit if no seeds exist
                resume_ransac = False
            else:
                planeIns, tmp_valid_mask, resume_ransac,  normals, center_coord, center_semseg, center_norm  =\
                    seq_ransac(coords, normals, semLab, center_prob_flat.clone(), planeIns, tmp_valid_mask , mask_surface.reshape([-1]),
                           norm_thres, radius,  area_thres, cur_planeId, vol_shape=center_prob.squeeze().shape, n_iter=500)

                if resume_ransac:
                    cur_planeId += 1
                    centers.append(center_coord)
                    center_segms.append(center_semseg)
                    center_norms.append(center_norm)


    return planeIns, normals,  centers, center_segms, center_norms


# ============== NN association ===========

def seq_ransac_NN(coords, normals, semLab, planeIns, valid_mask, mask_surface,
               norm_thres, radius, area_thres, cur_planeID,  vol_shape, n_iter=100):
    # sequential one-point plane ransac, the principle is the same as onePoint_ransan in fit_plane/util,
    # but the code is slightly different  because we do not have the instance level label,
    # so we need consider semantic label consistency here

    # sample the seeds w.r.t. their prob
    prob = valid_mask.float().clone()

    # in case replacement == False, and n_iter > (prob > 0).sum(), it will return idx whose weight == 0
    # therefore, we should make sure the n_iter is always < (prob>0).sum()
    idxs = torch.multinomial(prob, min(n_iter, (prob > 0).sum()), replacement=False)

    resume = False
    n_inliers = 0
    cnt_iter = 0
    best_mask = torch.zeros_like(semLab).type(torch.bool)
    # ransac iter
    for i in idxs:
        cnt_iter += 1
        sample_pnt = coords[:, i].unsqueeze(1)
        sample_norm = normals[:, i].unsqueeze(1)
        sample_semg = semLab[i]

        sys.stdout.write('\rprocessing: {}, assigning plane: {}, iter {}'.format(sample_semg.item(), cur_planeID, cnt_iter))
        sys.stdout.flush()

        # semantic should be same
        semseg_mask = (semLab == sample_semg)

        # normal should be similiar
        norm_mask = (torch.sum((normals * sample_norm), dim=0).abs() > norm_thres)

        # distance to the plane should under threshold
        planeD = (sample_norm * sample_pnt).sum()
        cluster_plane_dist = ((sample_norm * coords).sum(dim=0) - planeD).abs()
        spatial_mask = cluster_plane_dist <= radius

        # only sign once
        available_mask = (planeIns == 0) #| cur_planeID

        # connection mask -- only if the voxel have path connected to the sample pnt can be assigned
        cluster_mask = semseg_mask & norm_mask & spatial_mask & available_mask & mask_surface


        # put all the masks together --- Ideally we should put connection mask here, but it is too slow, we do it in post process instead
        # cluster_mask = cluster_mask & connection_mask

        n =  cluster_mask.sum()
        if n > n_inliers:
            best_mask = cluster_mask.clone()
            n_inliers = n
            center_idx, center_coord, center_semseg, center_norm = i, sample_pnt, sample_semg, sample_norm

    # best_mask = best_mask | (planeIns == cur_planeID) # for NN case
    # ransac will stop if the best plane_area < area_thres
    if n_inliers >= area_thres:
        print("{} assinged".format(cur_planeID))
        # break the mask if contain 2 seperate part
        connection_mask = check_connection(best_mask | valid_mask, center_idx, vol_shape)
        final_best_mask = connection_mask & best_mask

        # only update label if the area is sufficent large
        if final_best_mask.sum() < area_thres:
            return planeIns, valid_mask, resume, normals, sample_pnt, sample_semg, sample_norm, final_best_mask
        # ======= debug========
        # voxels = coords[:, mask_surface].T.cpu().numpy()
        # voxel_color = np.zeros([voxels.shape[0], 3])
        # voxel_color[best_mask.cpu().numpy()[mask_surface.cpu().numpy()]] =  np.array([0,0,255])
        #
        # pld = trimesh.points.PointCloud(vertices=voxels, colors=voxel_color, process=False)
        # pld.show()
        #
        # best_mask = connection_mask & best_mask
        # voxel_color[best_mask[mask_surface].cpu().numpy()] = np.array([255, 0, 0])
        # pld = trimesh.points.PointCloud(vertices=voxels, colors=voxel_color, process=False)
        # pld.show()


        planeIns[final_best_mask] = cur_planeID
        # planeIns[torch.logical_xor(final_best_mask, best_mask)] = 0 # the discard note connected (if itone set to be 0
        # valid_mask[final_best_mask] = True # add new added area for
        resume = True

        # get weird result -- perform least square fit in the ininlier -- we should not change normal, because the voxel norm if from tsdf_grad
        # it can be very different in the boundary
        mean_center = torch.mean(coords[:,best_mask].float(), dim=1, keepdim=True)
        dist = torch.sum((coords[:, best_mask].float() - mean_center).abs(), dim=0) # use l1 dist find nearest inliers to mean_center
        new_idx = dist.argmin(dim=0)  # acedend
        center_coord = coords[:,best_mask][:, new_idx].unsqueeze(1)
        normals[:, final_best_mask] = center_norm
    else:
        print("{} is too small, iliner {} not assinged".format(cur_planeID, n_inliers))
        center_coord, center_semseg, center_norm, final_best_mask = sample_pnt, sample_semg, sample_norm, best_mask # just return sth, will not be used

    return planeIns, valid_mask, resume, normals,  center_coord, center_semseg, center_norm, final_best_mask


def FloodFill(mask, normals, semLab, norm_thres, seed_idx, vol_shape):

    # set the seed
    candidate_mask = torch.zeros_like(mask).bool()
    candidate_mask[seed_idx] = True
    ctr_semg = semLab[seed_idx]
    # ctr_norm = normals[:, seed_idx]

    if ctr_semg in NONE_PLANE_ID:
        return candidate_mask.view(-1)

    candidate_mask = candidate_mask.view(vol_shape).unsqueeze(0).unsqueeze(0)

    pre_mask = candidate_mask.clone()

    # use 3D max pooling to propgate seed
    for cnt in range(max(vol_shape)): # longest dist to flood fill equals to the largest dim
        candidate_mask = F.max_pool3d(candidate_mask.float(), kernel_size=3, stride=1, padding=1).bool()

        # semantic should be same
        semseg_mask = (semLab == ctr_semg)
        # normal should be similiar
        # norm_mask = (torch.sum((normals * ctr_norm), dim=0).abs() > ORTHOGONAL_THRES) # same as gt generate
        # distance to the plane should under threshold -- emperical no difference

        candidate_mask = (candidate_mask.view(-1) & semseg_mask &  mask).view(vol_shape).unsqueeze(0).unsqueeze(0)

        if  (pre_mask == candidate_mask).all():
            break

        # new_mask = torch.logical_xor(pre_mask, candidate_mask) # the filled region in this iter
        pre_mask = candidate_mask.clone()

        # sys.stdout.write('\rflood fill step: {}'.format(cnt))
        # sys.stdout.flush()
    return candidate_mask.view(-1)


def get_center_FloodFill(coords, normals, semLab, seed_mask, norm_thres, idx_in_ori_idx, vol_shape):
    # center_prob is in vol_shape
    vaild_mask = seed_mask.clone()
    centers, center_segm_lst, center_norm_lst, center_ids, center_mask = [], [], [], [], []

    for id in idx_in_ori_idx:
        if vaild_mask[id]:
            cur_seed_mask = FloodFill(seed_mask, normals, semLab, norm_thres, id, vol_shape)
            if cur_seed_mask.sum() <= 4: continue # a instance center should have at least other 3 support

            mean_center = torch.mean(coords[:, cur_seed_mask].float(), dim=1, keepdim=True)
            dist = torch.sum((coords[:, cur_seed_mask].float() - mean_center).abs(), dim=0)

            new_idx = dist.argmin(dim=0)  # acedend

            # this normal update lead to wrong normal sometimes , but on average it leads to a bette results
            norm = normals[:, cur_seed_mask]
            if norm.shape[1] > 10000:   # to control the memory usage
                mask =  torch.randperm(norm.shape[1])[:10000]
                norm = norm[:, mask]
            u, s, v = torch.svd((norm.T))
            new_normal = v[:, 0].unsqueeze(1)
            seed_norm = normals[:, cur_seed_mask][:, new_idx].unsqueeze(1)
            new_normal = -new_normal if (new_normal*seed_norm).sum() < 0 else new_normal

            # we cannot use least square, some slice mask will turn to change their normal direction
            # new_normal, _ = torch.lstsq(torch.ones_like(coords[:1, cur_seed_mask]).float().T, coords[:, cur_seed_mask].float().T)
            # new_normal = new_normal[:coords.shape[0]]/new_normal[:coords.shape[0]].norm()

            centers.append(coords[:, cur_seed_mask][:, new_idx].unsqueeze(1))
            center_norm_lst.append(new_normal)
            # center_norm_lst.append(normals[:, cur_seed_mask][:, new_idx].unsqueeze(1))
            center_segm_lst.append(semLab[cur_seed_mask][new_idx])
            center_ids.append(id)
            center_mask.append(cur_seed_mask)

            vaild_mask[cur_seed_mask] = False

    return centers, center_norm_lst, center_segm_lst, center_ids, center_mask


def get_planeIns_NN(tsdf, cfg, ransac=False):
    # flood fill
    voxel_size = tsdf.voxel_size
    normals = tsdf.attribute_vols['plane_norm']
    semLab = tsdf.attribute_vols['semseg']
    plane_cls = tsdf.attribute_vols['plane_cls']
    center_prob = tsdf.attribute_vols['centroid_prob']

    radius = GROUPING_RADIUS[int(voxel_size * 100)]
    norm_thres = np.cos(np.deg2rad(cfg.MODEL.GROUPING.NORM_THRES))
    prob_thres = cfg.MODEL.GROUPING.PROB_THRES
    area_thres = PLANE_MIN_N_VOXELS[int(voxel_size * 100)]

    coords = coordinates(center_prob.shape, device=tsdf.device).type(torch.int16)

    normals, semLab, plane_cls = normals.reshape([3, -1]).type(torch.float), semLab.reshape([-1]), plane_cls.reshape([4, -1])#.type(coords.dtype)
    planeIns = torch.zeros_like(semLab).type(torch.int16)

    # pick valid center
    mask_surface = (tsdf.tsdf_vol.abs() < 1).reshape([-1]) & (plane_cls[0] > 0.5) #must be plane to be considered
    center_prob_flat = center_prob.clone().reshape([-1])
    seed_mask = (center_prob_flat > prob_thres) & mask_surface

    center_pred = (coords + plane_cls[1:].round().type(coords.dtype))

    if seed_mask.sum() == 0:
        return planeIns, normals, [], [], []

    # collect potential cluster center idx
    _, indices = torch.sort(center_prob_flat[seed_mask], descending=True)
    ori_indx = seed_mask.nonzero(as_tuple=False)
    idx_in_ori_idx = ori_indx[indices] # convert the valid set idx to the whole set idx

    # get center list
    centers, center_norms, center_segms, center_ids, center_masks = \
        get_center_FloodFill(coords, normals, semLab, seed_mask, norm_thres,  idx_in_ori_idx, center_prob.shape)

    n_ins = len(centers)
    # planeIns_distance = torch.ones([semLab.shape[0], n_ins], dtype=torch.int16).to(semLab.device).cpu() * 10000 #no gpu memory if use int64, if the center is hugh amount like scene 500_00 still not enough


    for i, (ctr_pnt, ctr_norm, ctr_semg, ctr_id, ctr_mask) in enumerate(zip(centers, center_norms, center_segms, center_ids, center_masks)):
        # semantic should be same
        cluster_mask = (semLab == ctr_semg) & mask_surface

        # normal should be similiar
        norm_mask = ((normals[:, cluster_mask] * ctr_norm).sum().abs() > norm_thres)
        cluster_mask[cluster_mask] &= norm_mask

        # distance to the plane should under threshold
        # planeD = (ctr_norm * ctr_pnt).sum()
        # cluster_plane_dist = ((coords * ctr_norm).sum(0) - planeD).abs()
        # spatial_mask = (cluster_plane_dist <= radius)
        # cluster_mask &= spatial_mask

       # the predicted center of the current candidates voxel is within the center support mask
        # https://stackoverflow.com/questions/41234161/check-common-elements-of-two-2d-numpy-arrays-either-row-or-column-wise
        cur_center_pred = center_pred[:, cluster_mask].T
        ctr_support_area = coords[:, ctr_mask].T
        # print(cur_center_pred.shape, ctr_support_area.shape, prob_thres)
        b_voted = (cur_center_pred[:, None] == ctr_support_area).all(-1).any(-1)
        cluster_mask[cluster_mask] &= b_voted

        # cluster_mask = check_connection(cluster_mask, ctr_id, center_prob.shape)
        planeIns[cluster_mask] = i +1

        # should associate to the cloest center instead of plane to preserve every instance
        # planeIns_distance[cluster_mask, i] =(coords[:,cluster_mask] - ctr_pnt).abs().sum(dim=0).type(torch.int16).cpu() # cluster_plane_dist[cluster_mask].type(torch.int16) #

    # discard very small instance
    if ransac:
        additional_id = 2
        for x in range(n_ins+1):
            cur_planeId = x + 1

            # merge unassigned voxels to exising instance first, and then generate plane in other unassigned region
            potentail_area_mask = (planeIns == cur_planeId) if x<n_ins else ((planeIns == 0) & mask_surface)  #center_mask[x] & (planeIns == 0)
            # print(n_ins)
            for semid in torch.unique(semLab[potentail_area_mask]): # for the planeIns case
                if semid.item() in NONE_PLANE_ID:continue
                resume_ransac = True
                tmp_valid_mask = potentail_area_mask & (semLab == semid)
                while resume_ransac:
                    if tmp_valid_mask.sum() == 0:  # quit if no seeds exist
                        resume_ransac = False
                    else:


                        planeIns, tmp_valid_mask, resume_ransac, normals, center_coord, center_semseg, center_norm, new_add_mask = \
                                seq_ransac_NN(coords, normals, semLab,  planeIns, tmp_valid_mask,
                                           mask_surface,
                                           norm_thres, radius, area_thres, cur_planeId, vol_shape=center_prob.squeeze().shape,
                                           n_iter=500)
                        # -------- debug -----------------
                        # if x == n_ins :
                        #     voxel_color[tmp_valid_mask[mask_surface.reshape([-1])].cpu().numpy()] = np.array([255, 0, 0])
                        #     voxel_color[new_add_mask[mask_surface.reshape([-1])].cpu().numpy()] = np.array([0, 255, 0])
                        #     print(resume_ransac)
                        #     pld = trimesh.points.PointCloud(vertices=voxels, colors=voxel_color, process=False)
                        #     pld.show()

                        # offer more id for the unassigned region
                        if cur_planeId >= n_ins + 1:
                            if resume_ransac:
                                centers.append(center_coord)
                                center_norms.append(center_norm)
                                center_segms.append(center_semseg)
                                tmp_valid_mask ^= new_add_mask

                            cur_planeId = n_ins + additional_id
                            additional_id += 1
                        else:
                            # enlarge merging seed
                            tmp_valid_mask |= new_add_mask



    for cur_planeId in torch.unique(planeIns):
        planeIns[planeIns == cur_planeId] = torch.where(planeIns[planeIns == cur_planeId].sum() < area_thres,
                                                        torch.zeros_like(planeIns[planeIns == cur_planeId]),
                                                        planeIns[planeIns == cur_planeId])


    return planeIns, normals, centers, center_segms, center_norms

def expl(val, ord=8, verbose = False):
    # approx the exp to speed up the process, as this part is done on cpu
    # idea from  https://codingforspeed.com/using-faster-exponential-approximation/
    # there is another brillant idea, but torch doest not support bit move >> or <<
    #  w.r.t. http://perso.citi-lab.fr/fdedinec/recherche/publis/2005-FPT.pdf
    x = 1. + val / (2**ord)
    for _ in range(ord):
        x *= x
        if verbose:
            print(x)
    return  x

def get_heatmap( planeIns_vol, voxel_size, tsdf_vol_sz):
    ## we expect flatten planeIns here
    # todo: This will limited only 1 batch in one GPU, change this to enable multiple batch
    coords_voxel = coordinates(tsdf_vol_sz[1:], device=planeIns_vol.device)
    centroid_vol = torch.zeros(planeIns_vol.view(-1).shape, device=planeIns_vol.device)

    # ====== build centroid heatmap =====
    unique_id = torch.unique(planeIns_vol)
    # centers = torch.zeros([3, len(unique_id)-1]).type(torch.long)
    for k, id in enumerate(unique_id):
        if id <= 0: continue  # none plane
        heatmap_mask = (planeIns_vol == id)  # ensure only update once
        cur_plane_voxels = coords_voxel[:, heatmap_mask]
        # cur_plane_norm = normal_vol[planeIns_vol == id, :]
        cur_center = torch.mean((cur_plane_voxels).type(torch.float), 1, keepdims=True)

        # if we use the mean center directly, it can fall into some empty voxel, then we cannot use its normal and depth
        # to generate plane proposal, therefore, we have to pick the one near to it and have the same plane normal
        dist = torch.sum((cur_plane_voxels - cur_center).abs(), dim=0)
        new_idx = torch.argmin(dist)
        new_center = cur_plane_voxels[:, new_idx].type(torch.long)

        dist = torch.sum((coords_voxel - new_center.unsqueeze(1).type(torch.float)) ** 2, dim=0)

        # plane ins id start from 1
        # only voxel belong to the instance can be update.
        # For a voxel does not have instance id, it normal will be different from plane norm, and so cannot be used to
        # propose the plane in the groupping stage

        std = (heatmap_mask.sum() * ADOPT_THRES + STD[voxel_size]) / 3
        # Gaussian_prob = torch.exp(-(dist / (2 * std))) / (torch.sqrt(std) * torch.sqrt(2 * PI))
        Gaussian_prob = expl(-(dist / (2 * std)).clamp(0, 500)) / ( torch.sqrt(std) * SQRT_2PI)  # clamp the dist for numerical stable
        Gaussian_prob[~heatmap_mask] = 0
        # change one of the max value to ensure only one center after approx
        max_val = Gaussian_prob.max()
        top_mask = (Gaussian_prob == max_val)

        if top_mask.sum() > 1:
            idx = top_mask.nonzero(as_tuple=False)
            Gaussian_prob[idx[0]] += 1e-4
        assert (Gaussian_prob == Gaussian_prob.max()).sum() == 1, "only one center for one instance"
        Gaussian_prob /= max_val  # normalize

        # only update the voxels have the same plane ins or those not belong to any plane once
        centroid_vol += Gaussian_prob
    return  centroid_vol.view(tsdf_vol_sz)



def coordinates(voxel_dim, device=torch.device('cuda'), b_flat=True):
    """ 3d meshgrid of given size.

    Args:
        voxel_dim: tuple of 3 ints (nx,ny,nz) specifying the size of the volume

    Returns:
        torch long tensor of size (3,nx*ny*nz)
    """

    nx, ny, nz = voxel_dim
    x = torch.arange(nx, dtype=torch.long, device=device)
    y = torch.arange(ny, dtype=torch.long, device=device)
    z = torch.arange(nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)
    if b_flat:
        return torch.stack((x.flatten(), y.flatten(), z.flatten()))
    else:
        return torch.stack((x, y, z))

def find_closet(src_arr, tgt_arr):
    # given 1D tgt_arr find closet val idx in src_arr
    # https://stackoverflow.com/questions/20780017/vectorize-finding-closest-value-in-an-array-for-each-element-in-another-array
    idx1 = torch.searchsorted(src_arr, tgt_arr).clamp(0, len(src_arr) -1)
    idx2 = (idx1 - 1).clamp(0, len(src_arr) - 1)

    diff1 = src_arr[idx1] - tgt_arr
    diff2 = tgt_arr - src_arr[idx2]

    final_idx = torch.where(diff1 <= diff2, idx1, idx2)
    return final_idx

def Eud2sphere(norm, d, rhos, thetas, phis):
    _rhos = d #param[3:, :]
    rhoIdxs = find_closet(rhos, _rhos)

    _thetas = torch.acos(norm[2]) #np.arccos(param[2:3, :])
    thetaIdxs = find_closet(thetas, _thetas)

    _phis = torch.acos(norm[0] / torch.sin(_thetas))
    phiIdxs = find_closet(phis, _phis)

    # ensure theta==0 only bring one activation
    if thetaIdxs == 0:
        phiIdxs = 0

    return rhoIdxs, thetaIdxs, phiIdxs

def sphere2norm(theta, phi):
    p_norm = torch.tensor([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)])
    return p_norm / p_norm.norm()
# ========================
# from vPlaneRecover.tsdf import SEM_INS_MAP

def get_planeInsVert_frmHT(tsdf, voxel_sz, origin, semLab_in, param_htmap_in, vote_idx,  rhos, thetas, phis, cfg):

    tsdf_vol = -tsdf.squeeze()

    # don't close surfaces using unknown-empty boundry
    tsdf_vol[tsdf_vol == -1] = 1

    tsdf_vol = tsdf_vol.clamp(-1, 1).cpu().numpy()

    semLab = semLab_in.squeeze()
    param_htmap = param_htmap_in.squeeze()

    verts_mc, faces, _, _ = measure.marching_cubes(tsdf_vol, level=0)
    verts_ind = np.round(verts_mc).astype(int)
    n_verts = verts_mc.shape[0]

    # in sone weird case ind will exceede the range
    d, h, w = semLab.shape
    if np.any(verts_ind[:, 0] > d - 1) or np.any(verts_ind[:, 1] > h - 1) or np.any(verts_ind[:, 2] > w - 1) or \
            np.any(verts_ind[:, 0] < 0) or np.any(verts_ind[:, 1] < 0) or np.any(verts_ind[:, 2] < 0):
        return trimesh.Trimesh(vertices=np.zeros((1, 3)))

    verts = verts_mc * voxel_sz + origin.cpu().numpy() # n*3
    verts_ind = np.round(verts_mc).astype(int)
    tmp_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    verts_norm = torch.from_numpy(tmp_mesh.vertex_normals).float().to(semLab.device) # n*3

    semseg_vol = semLab_in.detach().cpu().numpy()
    semseg_verts = semseg_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]

    voxel_coord_np = coordinates(semLab.shape, torch.device('cpu'), False).float().numpy() #semLab.device
    # voxel_coord_np = (voxel_coord_np - np.array([semLab.shape]).T / 2.).reshape([3, d, h, w])
    voxel_coord = torch.from_numpy(voxel_coord_np[:, verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]).float()
    voxel_coord = (voxel_coord -  torch.tensor([semLab.shape]).T / 2.).to(semLab.device)

    prob_thres = cfg.MODEL.GROUPING.PROB_THRES
    nms_r = 7
    planeIns = np.zeros_like(semseg_verts)
    ins_id = 1

    uniq_sem = (torch.unique(semLab))
    _param_htmap = F.threshold(param_htmap, prob_thres, 0)
    for sem_id in uniq_sem:
        # sys.stdout.write('\r process:{}'.format(CLASS_LABELS[sem_id.item()]))
        # sys.stdout.flush()
        if sem_id.item() not in SEM_INS_MAP:
            continue

        # load htmap and voxels under current semantic
        param_channel = SEM_INS_MAP[sem_id.item()]
        tmp = _param_htmap[param_channel]
        semseg_mask_np = (semseg_verts == sem_id.item())  # .view(-1)

        # nms
        nms_padding = (nms_r) // 2
        tmp_pool = F.max_pool3d(tmp.unsqueeze(0).unsqueeze(0), kernel_size=nms_r, stride=1,
                                padding=nms_padding).squeeze()
        tmp[tmp != tmp_pool] = 0
        params = tmp.nonzero(as_tuple=False)

        # assign
        h, w, d = semLab.shape
        rho_tol = (cfg.MODEL.BACKBONE3D.RHO_STEP[-1] +1) #if sem_id.item() in LAYOUT_SEM else (cfg.MODEL.BACKBONE3D.RHO_STEP[-1] +1)
        # componets = torch.arange(h*w*d).reshape([h,w,d]).to(semLab.device).float()
        norm_score_glb = torch.zeros([n_verts]).float().to(semLab.device)
        # tmp = torch.zeros_like(cur_param_htmap)
        for param_id in params:
            # tmp[param_id[0], param_id[1], param_id[2]] = 1.
            cur_norm = sphere2norm(thetas[param_id[1]], phis[param_id[2]]).float().to(semLab.device)
            cur_rho = torch.tensor(rhos[param_id[0]] * 2).float().to(semLab.device)

            norm_score = (cur_norm.view(1, 3) * verts_norm).sum(dim=1).abs()
            verts_norm_mask = norm_score > norm_score_glb

            verts_in_mask = ((voxel_coord.T @ cur_norm - cur_rho).abs()) <= rho_tol

            valid_pln_ins_mask = np.logical_and( (verts_in_mask & verts_norm_mask).detach().cpu().numpy(), semseg_mask_np )

            if valid_pln_ins_mask.sum() < 4:
                continue

            planeIns[valid_pln_ins_mask] = ins_id
            ins_id += 1
            norm_score_glb[valid_pln_ins_mask] = torch.where((norm_score_glb < norm_score)[valid_pln_ins_mask], norm_score[valid_pln_ins_mask],
                                                   norm_score_glb[valid_pln_ins_mask])

    return planeIns

def get_planeIns_htmap(tsdf, voxel_sz, origin, semLab_in, param_htmap_in, plane_norm, vote_idx,  rhos, thetas, phis, cfg, upscale=2):
    # voxel_size = tsdf.voxel_size
    valid_tsdf= tsdf.abs().squeeze() < 1
    semLab = semLab_in.squeeze()
    param_htmap = param_htmap_in.squeeze()

    verts_mc, faces, _, _ = measure.marching_cubes(tsdf, level=0)
    verts_ind = np.round(verts_mc).astype(int)

    verts = verts_mc * voxel_sz + origin.cpu().numpy()

    prob_thres = cfg.MODEL.GROUPING.PROB_THRES
    nms_r = 3
    voxel_coord = coordinates(semLab.shape, semLab.device).float()
    voxel_coord = voxel_coord - torch.tensor([semLab.shape]).to(semLab.device).T / 2.

    planeIns = torch.zeros_like(semLab).float()
    ins_id = 1
    uniq_sem = (torch.unique(semLab))

    _param_htmap = F.threshold(param_htmap, prob_thres, 0)

    for sem_id in uniq_sem:

        if sem_id.item() not in SEM_INS_MAP:
            continue

        # load htmap and voxels under current semantic
        param_channel = SEM_INS_MAP[sem_id.item()]
        tmp = _param_htmap[param_channel]
        cur_mask = (semLab == sem_id) #.view(-1)

        # nms
        nms_padding = (nms_r - 1) // 2
        tmp_pool = F.max_pool3d(tmp.unsqueeze(0).unsqueeze(0), kernel_size=nms_r, stride=1, padding=nms_padding).squeeze()
        tmp[tmp != tmp_pool] = 0
        params = tmp.nonzero(as_tuple=False)

        # assign
        valid_vlxs = valid_tsdf & cur_mask
        h, w, d = semLab.shape
        rho_tol = (cfg.MODEL.BACKBONE3D.RHO_STEP[-1] * 2 +1) if sem_id.item() in LAYOUT_SEM else (cfg.MODEL.BACKBONE3D.RHO_STEP[-1] +1)
        componets = torch.arange(h*w*d).reshape([h,w,d]).to(semLab.device).float()
        norm_score_glb = torch.zeros_like(semLab).float()

        for param_id in params:
            # tmp[param_id[0], param_id[1], param_id[2]] = 1.
            cur_norm = sphere2norm(thetas[param_id[1]], phis[param_id[2]]).float().to(semLab.device)
            cur_rho = torch.tensor(rhos[param_id[0]] * 2).float().to(semLab.device)

            planes_in_vol = ((voxel_coord.T @ cur_norm - cur_rho).abs()).reshape(semLab.shape)<= rho_tol

            norm_score = (cur_norm.view(3,1,1,1) * plane_norm).sum(dim=0).abs()
            norm_mask =  norm_score > norm_score_glb


            valid_planes_vlxs = norm_mask  &  planes_in_vol & valid_vlxs

            if valid_planes_vlxs.sum()<8:
                continue

            # connected components with floodfill
            comp = componets.clone()
            comp[~valid_planes_vlxs] = 0
            pre_comp = comp.clone()
            for cnt in range(max(semLab.shape)):  # longest dist to flood fill equals to the largest dim
                comp[valid_planes_vlxs] = F.max_pool3d(comp.unsqueeze(0).unsqueeze(0),
                                              kernel_size=3, stride=1, padding=1).squeeze()[valid_planes_vlxs]
                if (pre_comp == comp).all():
                    break
                else:
                    pre_comp = comp.clone()

            # assign label
            uniq_ins = torch.unique(comp)
            tmp_mask_sum = torch.zeros(len(uniq_ins))
            for i, tmp_id in enumerate(uniq_ins):
                if tmp_id < 1: continue
                tmp_mask = (comp == tmp_id)
                tmp_mask_sum[i] = tmp_mask.sum()

            tmp_thres = tmp_mask_sum.max() * 0.1
            for i, tmp_id in enumerate(uniq_ins):
                if tmp_id < 1: continue
                tmp_mask = (comp == tmp_id)
                if tmp_mask_sum[i] < tmp_thres: continue # filter extremely small ones
                planeIns[tmp_mask] = ins_id

                ins_id += 1

                # cancel the score if the voxel is not assigned
                norm_score_glb[tmp_mask] = torch.where((norm_score_glb < norm_score)[tmp_mask], norm_score[tmp_mask],
                                                         norm_score_glb[tmp_mask])

    return planeIns.long()
