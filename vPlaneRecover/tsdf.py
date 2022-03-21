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
from matplotlib.cm import get_cmap as colormap
import numpy as np
from skimage import measure
import torch
import trimesh

from vPlaneRecover.transforms import NYU40_COLORMAP
from vPlaneRecover.util import coordinates, Eud2sphere, SEM_INS_MAP, CLASS_LABELS

import matplotlib.colors as plcolors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt

ADOPT_THRES = 0.01
PI = np.pi
SQRT_2PI = np.sqrt(2 * PI)
INV_LN2 = 1 / np.log(2)
PROB_THRES = 0.5



def depth_to_world(projection, depth):
    """ backprojects depth maps to point clouds
    Args:
        projection: 3x4 projection matrix
        depth: hxw depth map

    Returns:
        tensor of 3d points 3x(h*w)
    """

    # add row to projection 3x4 -> 4x4
    eye_row = torch.tensor([[0,0,0,1]]).type_as(depth)
    projection = torch.cat((projection, eye_row))

    # pixel grid
    py, px = torch.meshgrid(torch.arange(depth.size(-2)).type_as(depth),
                            torch.arange(depth.size(-1)).type_as(depth))
    pz = torch.ones_like(px)
    p = torch.cat((px.unsqueeze(0), py.unsqueeze(0), pz.unsqueeze(0), 
                   1/depth.unsqueeze(0)))

    # backproject
    P = (projection.inverse() @ p.view(4,-1)).view(p.size())
    P = P[:3]/P[3:]
    return P


class TSDF():
    """ class to hold a truncated signed distance function (TSDF)

    Holds the TSDF volume along with meta data like voxel size and origin
    required to interpret the tsdf tensor.
    Also implements basic opperations on a TSDF like extracting a mesh.

    """

    def __init__(self, voxel_size, origin, tsdf_vol, attribute_vols=None,
                 attributes=None):
        """
        Args:
            voxel_size: metric size of voxels (ex: .04m)
            origin: origin of the voxel volume (xyz position of voxel (0,0,0))
            tsdf_vol: tensor of size hxwxd containing the TSDF values
            attribute_vols: dict of additional voxel volume data
                example: {'semseg':semseg} can be used to store a
                    semantic class id for each voxel
            attributes: dict of additional non voxel volume data (ex: instance
                labels, instance centers, ...)
        """

        # for plane HT
        self.rho_step = {0.04: 8, 0.08: 4, 0.16: 2}
        self.phis = torch.deg2rad(torch.arange(0.0, 180.0, 5))
        self.thetas = torch.deg2rad(torch.arange(0, 91.0, 90))
        self.n_tht, self.n_phi = len(self.thetas), len(self.phis)

        self.std = 1
        size = 6 * self.std + 3  # 1 unit larger than 3 sigma
        x = np.arange(0, size, 1, float)
        x0 = 3 * self.std + 1
        self.gaussian = torch.from_numpy(np.exp(- ((x - x0) ** 2 / (2 * self.std ** 2)))).float()

        self.voxel_size = voxel_size
        self.origin = origin
        self.tsdf_vol = tsdf_vol
        if attribute_vols is not None:
            self.attribute_vols = attribute_vols
        else:
            self.attribute_vols = {}
        if attributes is not None:
            self.attributes = attributes
        else:
            self.attributes = {}
        self.device = tsdf_vol.device

    def save(self, fname, addtional_info = None):
        data = {'origin': self.origin.cpu().numpy(),
                'voxel_size': self.voxel_size,
                'tsdf': self.tsdf_vol.detach().cpu().numpy()}
        for key, value in self.attribute_vols.items():
            # ignore the centers and segms plane_norm lists,  they should be pop during the vert_plane process,
            # but if it is not run or no plane is found, they will still be there
            if isinstance(value, list):continue
            if torch.is_tensor(value):
                data[key] = value.detach().cpu().numpy()
        for key, value in self.attributes.items():
            if torch.is_tensor(value):
                data[key] = value.cpu().numpy()

        if addtional_info is not None:
            for key in addtional_info:
                data[key] = addtional_info[key]

        np.savez_compressed(fname, **data)

    @classmethod
    def load(cls, fname, voxel_types=None):
        """ Load a tsdf from disk (stored as npz).

        Args:
            fname: path to archive
            voxel_types: list of strings specifying which volumes to load
                ex ['tsdf', 'color']. tsdf is loaded regardless.
                to load all volumes in archive use None (default)

        Returns:
            TSDF
        """

        with np.load(fname) as data:
            voxel_size = data['voxel_size'].item()
            origin = torch.as_tensor(data['origin']).view(1,3)
            tsdf_vol = torch.as_tensor(data['tsdf'])
            attribute_vols = {}
            attributes     = {}
            # the if here is to make it compa
            if 'color' in data and (voxel_types is None or 'color' in voxel_types):
                attribute_vols['color'] = torch.as_tensor(data['color'])

            if ('instance' in data and (voxel_types is None or
                                        'instance' in voxel_types or
                                        'semseg' in voxel_types)):
                attribute_vols['instance'] = torch.as_tensor(data['instance'])

            if voxel_types is None or 'plane_ins' in voxel_types:
                if ('plane_ins') in data:
                    attribute_vols['plane_ins'] = torch.as_tensor(data['plane_ins']).type(torch.long)

                if ('plane_cls') in data:
                    attribute_vols['plane_cls'] = torch.as_tensor(data['plane_cls']).type(torch.long)

                if 'plane_id2param' in data:
                    tmp = torch.as_tensor(data['plane_id2param']).type(torch.float) #n*3
                    # convert to [n, -d] version for transformation connvience
                    n_param, param_dim = tmp.shape
                    if param_dim == 3:
                        tmpD = tmp.norm(dim=1, keepdim=True)
                        tmpN = tmp / tmpD
                        _param = torch.cat([tmpN, -tmpD], dim=1)
                    else:
                        _param = tmp
                    if n_param < 1000: #No scene has more than 1000 planes in the dataset, change it to bigger one if needed
                        _param = torch.cat([_param, torch.zeros([1000-n_param, 4])], 0) # pad param to make sure every sample in the same shape for batch reading

                    attribute_vols['plane_id2param'] = _param

            if ('centroid_prob') in data and (voxel_types is None or
                                          'centroid_prob' in voxel_types or
                                            'cenprob' in voxel_types) and 'plane_ins' in data:

                # the random transform will bias the centroid, we have to generate it online
                # attribute_vols['centroid_prob'] = torch.as_tensor(data['centroid_prob']).type(torch.float)
                attribute_vols['plane_ins'] = torch.as_tensor(data['plane_ins']).type(torch.long)


            ret = cls(voxel_size, origin, tsdf_vol, attribute_vols, attributes)
        return ret

    def to(self, device):
        """ Move tensors to a device"""

        self.origin = self.origin.to(device)
        self.tsdf_vol = self.tsdf_vol.to(device)
        self.attribute_vols = {key:value.to(device)
                               for key, value in self.attribute_vols.items()}
        self.attributes = {key:value.to(device)
                           for key, value in self.attributes.items()}
        self.device = device
        return self

    def get_planeColor_map(self, n_plane):
        # get color
        _cmap = np.array(NYU40_COLORMAP)
        if n_plane - 41 > 0:
            cmap = (colormap('jet')(np.linspace(0, 1, n_plane - 41))[:, :3] * 255).astype(np.uint8)
            cmap = cmap[np.random.permutation(n_plane - 41), :]
            plane_color = np.concatenate([_cmap, cmap], axis=0)
        else:
            plane_color = _cmap

        return plane_color

    def get_mesh(self, attribute='color', cmap='nyu40'):
        """ Extract a mesh from the TSDF using marching cubes

        If TSDF also has atribute_vols, these are extracted as
        vertex_attributes. The mesh is also colored using the cmap

        Args:
            attribute: which tsdf attribute is used to color the mesh
            cmap: colormap for converting the attribute to a color

        Returns:
            trimesh.Trimesh
        """

        tsdf_vol = self.tsdf_vol.detach().clone()

        # measure.marching_cubes() likes positive
        # values in front of surface
        tsdf_vol = -tsdf_vol

        # don't close surfaces using unknown-empty boundry
        tsdf_vol[tsdf_vol==-1]=1

        tsdf_vol = tsdf_vol.clamp(-1,1).cpu().numpy()

        if tsdf_vol.min()>=0 or tsdf_vol.max()<=0:
            return trimesh.Trimesh(vertices=np.zeros((1,3)))

        try:
            # still meet an error say "RuntimeError: No surface found at the given iso value"
            # todo: figure out why
            verts_mc, faces, _, _ = measure.marching_cubes(tsdf_vol, level=0)
        except:
            return trimesh.Trimesh(vertices=np.zeros((1, 3)))

        verts_ind = np.round(verts_mc).astype(int)
        # in sone weird case ind will exceede the range
        d, h, w = tsdf_vol.shape
        if np.any(verts_ind[:,0] > d-1) or np.any(verts_ind[:,1]>h-1) or  np.any(verts_ind[:,2] > w-1) or \
            np.any(verts_ind[:,0] < 0) or np.any(verts_ind[:,1]<0) or  np.any(verts_ind[:,2] < 0):
            return trimesh.Trimesh(vertices=np.zeros((1, 3)))

        verts = verts_mc * self.voxel_size + self.origin.cpu().numpy()

        vertex_attributes = {}
        # get vertex attributes
        if 'semseg' in self.attribute_vols:
            semseg_vol = self.attribute_vols['semseg'].detach().cpu().numpy()
            semseg = semseg_vol[verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]] # the semantic label is copy from the voxel label whre the verts located
            vertex_attributes['semseg'] = semseg

        if 'instance' in self.attribute_vols:
            instance_vol = self.attribute_vols['instance']
            instance_vol = instance_vol.detach().cpu().numpy()
            instance = instance_vol[verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]]
            vertex_attributes['instance'] = instance

        norm_vol = None
        if 'plane_norm' in self.attribute_vols:
            norm_vol = self.attribute_vols['plane_norm']
            norm_vol = norm_vol.detach().cpu().numpy()
            norm_vol = norm_vol[:, verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]].T

        if 'plane_cls' in self.attribute_vols:
            planeCls_vol = self.attribute_vols['plane_cls'].detach().cpu().numpy()
            planeCls_vol = planeCls_vol[:, verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
            # vertex_attributes['plane_cls'] = planeCls_vol

        if 'plane_ins' in self.attribute_vols:
            planeIns_vol = self.attribute_vols['plane_ins'].detach().cpu().numpy()
            planeIns_vol = planeIns_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
            vertex_attributes['plane_ins'] = planeIns_vol

        elif 'plane_ins_vert'  in self.attribute_vols:
            planeIns_vol = self.attribute_vols['plane_ins_vert']
            vertex_attributes['plane_ins'] = planeIns_vol

        if 'parts' in self.attribute_vols:
            parts_vol = self.attribute_vols['parts'].detach().cpu().numpy()
            parts_vol = parts_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
            # vertex_attributes['parts'] = parts_vol

        # to run mc once
        meshes = {}
        # color mesh
        if 'color'  in attribute and 'color' in self.attribute_vols:
            color_vol = self.attribute_vols['color']
            color_vol = color_vol.detach().clamp(0,255).byte().cpu().numpy()
            colors = color_vol[:, verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]].T
            meshes['color_mesh'] = trimesh.Trimesh(
                                    vertices=verts, faces=faces, vertex_colors=colors, process=False)
        if 'instance' in attribute:
            label_viz = instance+1
            n=label_viz.max()
            cmap = (colormap('jet')(np.linspace(0,1,n))[:,:3]*255).astype(np.uint8)
            cmap = cmap[np.random.permutation(n),:]
            cmap = np.insert(cmap,0,[0,0,0],0)
            colors = cmap[label_viz,:]
            meshes['ins_mesh'] =  trimesh.Trimesh(
                                    vertices=verts, faces=faces, vertex_colors=colors,
                                    vertex_attributes=vertex_attributes, process=False)


        if 'semseg' in attribute and ('semseg' in self.attribute_vols):
            if cmap=='nyu40':
                cmap = np.array(NYU40_COLORMAP) # FIXME: support more general colormaps
            else:
                raise NotImplementedError('colormap %s'%cmap)
            label_viz = semseg.copy()
            label_viz[(label_viz<0) | (label_viz>=len(cmap))]=0
            colors = cmap[label_viz,:]
            meshes['semseg'] =  trimesh.Trimesh(
                                    vertices=verts, faces=faces, vertex_colors=colors,vertex_normals=norm_vol,
                                    vertex_attributes=vertex_attributes, process=False)

            if 'semseg_ent' in self.attribute_vols:
                ent = self.attribute_vols['semseg_ent'][verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]].detach().cpu().numpy()
                jet = plt.get_cmap('jet')
                cNorm = plcolors.Normalize(vmin=0, vmax=3)
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
                colors = scalarMap.to_rgba(ent.flatten())
                meshes['semseg_ent'] = trimesh.Trimesh(
                    vertices=verts, faces=faces, vertex_colors=colors, process=False)

        if  'normal' in attribute and 'plane_norm' in self.attribute_vols:
            colors = (norm_vol + 1) / 2
            meshes['norm'] =trimesh.Trimesh(
                            vertices=verts, faces=faces, vertex_normals=norm_vol, vertex_colors=colors, process=False)

        if 'plane_cls' in attribute  and ('plane_cls' in self.attribute_vols):
            viz = planeCls_vol[1:].copy().clip(-25, 25).T
            colors = (viz + 25.) / 50.
            colors[planeCls_vol[0] ==0] = 0
            meshes['plane_cls'] = trimesh.Trimesh(
                vertices=verts, faces=faces, vertex_colors=colors, process=False)

        if ('plane_ins_vert'  in self.attribute_vols) or ('plane_ins' in self.attribute_vols):
            n_plane = planeIns_vol.max() + 1 # is not equal to the real pred plane num, but the largest plane idx

            # if no center is detected
            if n_plane == 0:
                meshes['plane_ins'] = trimesh.Trimesh(vertices=np.zeros((1, 3)))
            else:
                plane_color = self.get_planeColor_map(n_plane)
                colors = plane_color[planeIns_vol,:]

                meshes['plane_ins'] = trimesh.Trimesh(
                    vertices=verts, faces=faces, vertex_colors=colors, process=False)

                # introduce regularity
                new_verts = verts.copy()
                new_vert_norms =  meshes['plane_ins'].vertex_normals

                face_color =  meshes['plane_ins'].visual.face_colors[:,:3]
                pln_ins = np.unique(planeIns_vol)
                for k, id in enumerate(pln_ins):
                    mask =  planeIns_vol == id
                    cur_verts = new_verts[mask]

                proj_mesh = trimesh.Trimesh(
                    vertices=new_verts, faces=faces, vertex_colors=colors,  process=False)


        if 'centroid_prob' in attribute and ('centroid_prob' in self.attribute_vols):
            centroid_prob = self.attribute_vols['centroid_prob'].detach().cpu().numpy()
            cen_prob = centroid_prob[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
            jet = plt.get_cmap('jet')
            cNorm = plcolors.Normalize(vmin=0, vmax=1)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
            colors = scalarMap.to_rgba(cen_prob.flatten())
            meshes['centroid_prob'] = trimesh.Trimesh(
                vertices=verts, faces=faces, vertex_colors=colors,  process=False)

            voxel_coords = coordinates(self.tsdf_vol.size(), device=torch.device('cpu'))#.numpy().T
            voxel_mask = (centroid_prob > PROB_THRES).flatten()

            if voxel_mask.sum() == 0:
                meshes['centroid_pld'] = trimesh.points.PointCloud(vertices=np.zeros((1, 3)), colors=np.zeros((1,3)))
            else:
                viz_pnts = (voxel_coords[:, voxel_mask].T * self.voxel_size + self.origin.cpu()).numpy()
                pld = trimesh.points.PointCloud(vertices=viz_pnts,
                                                colors=scalarMap.to_rgba(centroid_prob.flatten())[voxel_mask], process=False)
                meshes['centroid_pld'] = pld

        if 'vert_plane' in attribute and 'plane_ins' in self.attribute_vols and 'centers' in self.attribute_vols:

            if len(self.attribute_vols['centers']) == 0:
                meshes['vert_plane'] = trimesh.Trimesh(
                    vertices=verts, faces=faces, vertex_colors=colors, process=False)
            else:
                res_verts = verts_mc.copy()
                plane_vert_mask = np.zeros(verts_mc.shape[0]).astype(np.bool)
                verts_planeIns = planeIns_vol.copy()
                centers, center_segms, center_norms = self.attribute_vols.pop('centers'), \
                                                       self.attribute_vols.pop('center_segms'), \
                                                        self.attribute_vols.pop('center_norms')
                for i, (center_coord, center_seg, center_norm) in enumerate(zip(centers, center_segms, center_norms)):
                    # select the verts fall into the current plane voxel
                    vert_mask = (verts_planeIns == (i + 1))

                    if vert_mask.sum() == 0: continue # some center may not associate to a plane if no verts fall into the voxels

                    cluster_verts = verts_mc[vert_mask, :]

                    # proj these verts to the plane,
                    planeD = (center_norm * center_coord).sum().detach().cpu().numpy()
                    cluster_plane_dist = (center_norm.T.detach().cpu().numpy()* cluster_verts).sum(axis=1) - planeD
                    dist_sign = np.sign(cluster_plane_dist)
                    proj_verts = cluster_verts - (dist_sign[:,np.newaxis]* #.unsqueeze(1)
                                                  np.matmul(np.abs(cluster_plane_dist)[:,np.newaxis],
                                                               center_norm.T.detach().cpu().numpy()))

                    res_verts[vert_mask] = proj_verts
                    plane_vert_mask |= vert_mask

                res_verts = res_verts * self.voxel_size + self.origin.cpu().numpy()
                plane_color = self.get_planeColor_map(n_plane)
                colors = plane_color[verts_planeIns, :]

                meshes['vert_plane'] = trimesh.Trimesh(
                    vertices=res_verts, faces=faces, vertex_normals=norm_vol, vertex_colors=colors, process=False)

                # filter non-plane verts
                res_faces = self.filter_plane_vert(plane_vert_mask, faces)
                mask = res_faces.sum(1) > 0
                meshes['vert_plane_filter'] = trimesh.Trimesh(
                    vertices=res_verts, faces=res_faces[mask], vertex_normals=norm_vol, vertex_colors=colors, process=False)

        if 'parts' in attribute and ('parts' in self.attribute_vols):
            viz = parts_vol.copy()
            colors = np.stack([viz==0 , viz==1 , viz==2 ], axis=1).astype(np.float)*255
            meshes['parts'] = trimesh.Trimesh(
                vertices=verts, faces=faces, vertex_colors=colors, process=False)

        if 'param_htmap' in attribute and 'param_htmap' in self.attribute_vols:
            n_map = self.attribute_vols['param_htmap'].shape[0]
            htmap = self.attribute_vols['param_htmap']#.squeeze()

            param_coord_dim = htmap.shape[-3:]
            param_verts = coordinates(param_coord_dim, device='cpu').T.numpy()

            for i in range(n_map):
                jet = plt.get_cmap('jet')
                cNorm = plcolors.Normalize(vmin=0, vmax=1)
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
                cur_htmap = htmap[i]
                meshes['param_htmap_{}'.format(CLASS_LABELS[i])] = trimesh.points.PointCloud(vertices=param_verts,
                                                      colors=scalarMap.to_rgba(cur_htmap.view(-1,1).detach().cpu().numpy().flatten()), process=False)


        if attribute == 'eval' :
            colors = None
            meshes['eval'] = trimesh.Trimesh(
                    vertices=verts, faces=faces, vertex_colors=colors, process=False)

        return meshes

    def filter_plane_vert(self, plane_vert_mask, faces):
        non_plane_verts = (~plane_vert_mask).nonzero()[0]
        res_faces = faces.copy()
        for r in range(faces.shape[0]):
            cnt = 0
            for i in range(3):
                if res_faces[r, i] in non_plane_verts:
                    cnt += 1
            if cnt > 0 and cnt < 3:  # filter the face which has both plane and non-plane verts
                res_faces[r] = -1

        return res_faces

    def expl(self, val, ord=8, verbose = False):
        # approx the exp to speed up the process, as this part is done on cpu
        # idea from  https://codingforspeed.com/using-faster-exponential-approximation/
        # there is another brillant idea, but torch doest not support bit move >> or <<
        #  w.r.t. http://perso.citi-lab.fr/fdedinec/recherche/publis/2005-FPT.pdf
        x = 1. + val / (2**ord)
        for _ in range(ord):
            x = x * x
            if verbose:
                print(x)
        return  x


    def get_paramHtmap(self, planeIns_vol_in, semseg_vol, plane_id2param, coords_voxel):


        voxel_sz = self.voxel_size

        # diag range
        row, col, hght = planeIns_vol_in.shape
        coord_ctrlz = coords_voxel - torch.tensor([row//2, col//2, hght//2]).to(coords_voxel.device).unsqueeze(1)

        diag = np.maximum(np.sqrt((row - 1) ** 2 + (col - 1) ** 2), hght)
        q = np.ceil(diag / self.rho_step[voxel_sz])
        n_rho = int(q + 1)  # int(2 * q + 1)
        rhos = torch.linspace(-q // 2 * self.rho_step[voxel_sz], q // 2 * self.rho_step[voxel_sz], n_rho)


        # _HT_map = torch.zeros([1, n_rho, self.n_tht, self.n_phi]).to(planeIns_vol_in.device)
        uniq_ins = (torch.unique(planeIns_vol_in))
        n_htmap = len(SEM_INS_MAP)
        HT_map = torch.zeros([n_htmap, n_rho, self.n_tht, self.n_phi]).to(planeIns_vol_in.device)


        for label in uniq_ins:
            if label < 1: continue
            planeIns_mask = (planeIns_vol_in == label).view(-1)

            # only care about the eval class currently
            cur_sem =  torch.mode(semseg_vol.view(-1)[planeIns_mask])[0] # not in [1, 2, 22]:continue
            if cur_sem.item() not in SEM_INS_MAP:
                continue
            cur_sem = SEM_INS_MAP[cur_sem.item()]

            voxel_norm = plane_id2param[:3, label]
            if voxel_norm[2].abs() > 0.05 and voxel_norm[2].abs() < 0.95:
                # does not follow our assumption  0 or 1, then enfore to be fit with a close one
                # todo this constrain should be removed if we use more than 2 theta_step
                voxel_norm[2] = voxel_norm[2].sign() * torch.ceil(voxel_norm[2].abs() - 0.6)
                voxel_norm /= voxel_norm.norm()

            mean_pnt = coord_ctrlz[:, planeIns_mask].float().mean(dim=1)
            voxel_d = mean_pnt @ voxel_norm

            cm_vec = mean_pnt / mean_pnt.norm()
            cy_vec = torch.tensor([0, 1, 0]).to(mean_pnt.device)
            direc = (cm_vec * cy_vec).sum() # determine if plane in y-pos or y-neg

            # in y-pos, we ensure norm point away from origin  -->  change if (direc > 0 and voxel_d < 0)
            # in  y-neg, we ensure norm point away from origin -->  change if (direc < 0 and voxel_d > 0)
            if direc * voxel_d < 0: #(direc < 0 and voxel_d > 0) or (direc > 0 and voxel_d < 0)
                voxel_norm *= -1
                voxel_d *= -1

            # further ensure  voxel_norm[1] is pos, if it is a vertical plane,  according to our phi definiation
            # if it is a horizontal plane,, we ensure  voxel_norm[2] > 0, according to our theta defination
            if  voxel_norm[2].abs() < 0.05:
                if voxel_norm[1] < 0 :
                    voxel_norm *= -1
                    voxel_d *= -1
            else:
                if voxel_norm[2] < 0:
                    voxel_norm *= -1
                    voxel_d *= -1

            rhoIdx, thetaIdx, phiIdx = Eud2sphere(voxel_norm, voxel_d, rhos, self.thetas, self.phis)
            if thetaIdx == 0 and phiIdx == 0:
                rhoIdx_up, rhoIdx_down = (rhos==hght//2).nonzero(as_tuple=False)[0,0], (rhos==-hght//2).nonzero(as_tuple=False)[0,0]
                rhoIdx = rhoIdx.clamp(rhoIdx_down, rhoIdx_up)
            # build guassian heatmap --- cannot do this because some instance are in same plane
            # the following code will make this surrounding all 1
            if HT_map[cur_sem, rhoIdx, thetaIdx, phiIdx] >= 1: continue

            rho_l, rho_r = (rhoIdx - 3 * self.std - 1), (rhoIdx + 3 * self.std + 2)
            ga_rl, ga_rr = max(0, -rho_l), min(rho_r, n_rho ) -  rho_l
            rho_x, rho_y = max(0, rho_l), min(rho_r,n_rho)
            if thetaIdx == 0:
                # for horizontal plane, guassian on rho only
                HT_map[cur_sem, rho_x:rho_y, thetaIdx, phiIdx] = torch.max(HT_map[cur_sem, rho_x:rho_y, thetaIdx, phiIdx] ,
                                                                     self.gaussian[ga_rl: ga_rr])
            else:
                HT_map[cur_sem, rho_x:rho_y, thetaIdx, phiIdx] =  torch.max(HT_map[cur_sem, rho_x:rho_y, thetaIdx, phiIdx] ,
                                                                     self.gaussian[ga_rl: ga_rr])

                phi_l, phi_r = (phiIdx - 3 * self.std - 1), (phiIdx + 3 * self.std + 2)
                ga_pl, ga_pr = max(0, -phi_l), min(phi_r, self.n_phi) - phi_l
                phi_x, phi_y = max(phi_l, 0), min(phi_r, self.n_phi)
                HT_map[cur_sem, rhoIdx, thetaIdx, phi_x:phi_y] = torch.max(  HT_map[cur_sem, rhoIdx, thetaIdx, phi_x:phi_y] ,
                                                                       self.gaussian[ga_pl:ga_pr])


        return HT_map, coord_ctrlz

    def get_heatmap(self, planeIns_vol_in, coords_voxel):

        centroid_vol = torch.zeros(planeIns_vol_in.view(-1).shape)
        planeIns_vol = planeIns_vol_in.view(-1)

        planeCls_vol = torch.zeros([4, centroid_vol.shape[0]]).long()
        planeCls_vol[0, planeIns_vol > 0] = 1

        # ====== build centroid heatmap =====
        unique_id = torch.unique(planeIns_vol)
        for k, id in enumerate(unique_id):
            if id == 0: continue  # none plane
            heatmap_mask = (planeIns_vol == id)
            cur_plane_voxels = coords_voxel[:, heatmap_mask]
            # cur_plane_norm = normal_vol[planeIns_vol == id, :]
            cur_center, _ = torch.median((cur_plane_voxels).type(torch.float), 1, keepdims=True)
            new_center = cur_center

            dist = torch.sum((coords_voxel - new_center.unsqueeze(1).type(torch.float)) ** 2, dim=0)

            # plane ins id start from 1
            # only voxel belong to the instance can be update.
            std = (heatmap_mask.sum() * ADOPT_THRES )  #+ STD[voxel_size]
            # Gaussian_prob = torch.exp(-(dist / (2 * std))) / (np.sqrt(std) * np.sqrt(2 * PI))
            Gaussian_prob = self.expl(-(dist / (2 * std)).clamp(0,500)) / (np.sqrt(std) * SQRT_2PI) # clamp the dist for numerical stable
            Gaussian_prob[~heatmap_mask] = 0

            # change one of the max value to ensure only one center after approx
            max_val = Gaussian_prob.max()
            top_mask = (Gaussian_prob == max_val)

            if top_mask.sum() > 1:
                idx = top_mask.nonzero(as_tuple=False)
                Gaussian_prob[idx[0]] += 1e-4
            assert (Gaussian_prob == Gaussian_prob.max()).sum() == 1, "only one center for one instance"
            Gaussian_prob /= max_val # normalize

            # only update the voxels have the same plane ins or those not belong to any plane once
            centroid_vol += Gaussian_prob
            planeCls_vol[1:, heatmap_mask] = new_center.unsqueeze(1) - cur_plane_voxels

        return  centroid_vol.view(planeIns_vol_in.shape), planeCls_vol.view([4] + list(planeIns_vol_in.shape))

    def transform(self, transform=None, voxel_dim=None, origin=None,
                  align_corners=False):
        """ Applies a 3x4 linear transformation to the TSDF.

        Each voxel is moved according to the transformation and a new volume
        is constructed with the result.

        Args:
            transform: 3x4 linear transform
            voxel_dim: size of output voxel volume to construct (nx,ny,nz)
                default (None) is the same size as the input
            origin: origin of the voxel volume (xyz position of voxel (0,0,0))
                default (None) is the same as the input

        Returns:
            A new TSDF with the transformed coordinates
        """

        device = self.tsdf_vol.device

        old_voxel_dim = list(self.tsdf_vol.size())
        old_origin = self.origin

        if transform is None:
            transform = torch.eye(4, device=device)
        if voxel_dim is None:
            voxel_dim = old_voxel_dim
        if origin is None:
            origin = old_origin
        else:
            origin = torch.tensor(origin, dtype=torch.float, device=device).view(1,3)
        # print('transform', origin)
        coords_voxel = coordinates(voxel_dim, device) #coordinate of out size
        world = coords_voxel.type(torch.float) * self.voxel_size + origin.T
        world = torch.cat((world, torch.ones_like(world[:1]) ), dim=0)
        world = transform[:3,:] @ world
        coords = (world - old_origin.T) / self.voxel_size

        # grid sample expects coords in [-1,1]
        coords = 2*coords/(torch.tensor(old_voxel_dim, device=device)-1).view(3,1)-1 # move the origin to the center
        coords = coords[[2,1,0]].T.view([1]+voxel_dim+[3]) # (z, y, x) for each voxel

        # bilinear interpolation near surface,
        # no interpolation along -1,1 boundry
        tsdf_vol = torch.nn.functional.grid_sample(
            self.tsdf_vol.view([1,1]+old_voxel_dim),
            coords, mode='nearest', align_corners=align_corners
        ).squeeze()
        tsdf_vol_bilin = torch.nn.functional.grid_sample(
            self.tsdf_vol.view([1,1]+old_voxel_dim), coords, mode='bilinear',
            align_corners=align_corners
        ).squeeze()
        mask = tsdf_vol.abs()<1
        tsdf_vol[mask] = tsdf_vol_bilin[mask]

        # padding_mode='ones' does not exist for grid_sample so replace
        # elements that were on the boarder with 1.
        # voxels beyond full volume (prior to croping) should be marked as empty
        mask = (coords.abs()>=1).squeeze(0).any(3)
        tsdf_vol[mask] = 1

        # transform attribute_vols
        attribute_vols={}
        for key, value in self.attribute_vols.items():
            if key == 'plane_id2param':
                param = transform[:3, :3].T @ value.T[:3]
                attribute_vols[key] = param
                continue

            dtype = value.dtype
            if len(value.size())==3:
                channels=1
            else:
                channels=value.size(0)
            value = value.view([1,channels]+old_voxel_dim).float()
            mode = 'bilinear' if dtype==torch.float else 'nearest'
            attribute_vols[key] = torch.nn.functional.grid_sample(
                value, coords, mode=mode, align_corners=align_corners
            ).squeeze().type(dtype)

            if key=='mask_outside':
                attribute_vols[key][mask] = True
            elif key=='semseg':
                attribute_vols[key][mask] = -1
            # elif key == 'centroid_prob':
            #     attribute_vols[key][mask] = -1
            elif key == 'plane_norm':
                attribute_vols[key][:, mask] = 0
            # elif key == 'plane_cls':
            #     attribute_vols[key][:, mask] = -1
            elif key == 'plane_ins':
                # compute centroid prob and plane cls online, due to the data augmentation
                # attribute_vols['centroid_prob'], attribute_vols['plane_cls'] = self.get_heatmap(attribute_vols[key],
                #                                                                                 coords_voxel)
                attribute_vols[key][mask] = -1
                # TODO: transform attributes

        # build hough htmap
        if 'plane_ins' in attribute_vols and 'plane_id2param' in attribute_vols:
            attribute_vols['param_htmap'], _ = self.get_paramHtmap(
                attribute_vols['plane_ins'],
                attribute_vols['semseg'],
                attribute_vols['plane_id2param'],
                coords_voxel)   # attribute_vols['hough_coords']

        attributes = self.attributes

        # cropping by return a new tsdf
        return TSDF(self.voxel_size, origin, tsdf_vol, attribute_vols, attributes)



class TSDFFusion():
    """ Accumulates depth maps into a TSDF using TSDF fusion"""

    def __init__(self, voxel_dim=(128,128,128), voxel_size=.02, origin=(0,0,0),
                 trunc_ratio=3, device=torch.device('cuda'),
                 color=True, label=False):
        """
        Args:
            voxel_dim: tuple of 3 ints (nx,ny,nz) specifying the size of the volume
            voxel_size: metric size of each voxel (ex: .04m)
            origin: origin of the voxel volume (xyz position of voxel (0,0,0))
            trunc_ratio: number of voxels before truncating to 1
            device: cpu/gpu
            color: if True an RGB color volume is also accumulated
            label: if True a semantic/instance label volume is also accumulated
        """
        nx, ny, nz = voxel_dim
        self.voxel_dim = voxel_dim
        self.voxel_size = voxel_size # like unit for each coordinate, 1 coordinate means 4cm here
        self.origin = torch.tensor(origin, dtype=torch.float, device=device).view(1,3)
        self.trunc_margin = voxel_size * trunc_ratio
        self.device = device
        coords = coordinates(voxel_dim, device)
        world = coords.type(torch.float) * voxel_size + self.origin.T # note the origin changed scene by scene according to point distribution
        self.world = torch.cat((world, torch.ones_like(world[:1]) ), dim=0)

        self.tsdf_vol = torch.ones(nx*ny*nz, device=device)
        self.weight_vol = torch.zeros(nx*ny*nz, device=device)

        if color:
            self.color_vol = torch.zeros((3,nx*ny*nz), device=device)
        else:
            self.color_vol = None

        if label:
            self.label_vol = -torch.ones(nx*ny*nz, device=device, dtype=torch.long)
        else:
            self.label_vol = None

    def reset(self):
        """ Initialize the volumes to default values"""

        self.tsdf_vol.fill_(1)
        self.weight_vol.fill_(0)
        if self.color_vol is not None:
            self.color_vol.fill_(0)
        if self.label_vol is not None:
             self.label_vol.fill_(-1)

    def integrate(self, projection, depth, color=None, label=None):
        """ Accumulate a depth map (and color/label) into the TSDF

        Args:
            projection: projection matrix of the camera (intrinsics@extrinsics)
            depth: hxw depth map
            color: 3xhxw RGB image
            label: hxw label map
        """

        # world coords to camera coords, self.world is the voxelized world coord, w.r.t. the adopted origin
        camera = projection @ self.world # (3 x 4) @ (4 x (nx*ny*nz))
        px = (camera[0,:]/camera[2,:]).round().type(torch.long) # (nx*ny*nz) , pixel coord x for each voxel in cur_camera
        py = (camera[1,:]/camera[2,:]).round().type(torch.long) # (nx*ny*nz)
        pz = camera[2,:] #(nx*ny*nz)

        # voxels in view frustrum
        height, width = depth.size()
        valid = (px >= 0) & (py >= 0) & (px < width) & (py < height) & (pz>0) # (nx*ny*nz)

        # voxels with valid depth
        valid[valid.clone()]  *= (depth[py[valid.clone()], px[valid.clone()]]>0)

        # tsdf distance, tsdf = max(-1, min(1, sdf/t)), t is the truncated value, because far away value usually not related to surface reconstruction
        # please refer to tsdf:experiment on voxel size  (a.alhamadi 2014) for detail
        dist = pz[valid] - depth[py[valid], px[valid]] # (n1) (where n1 is # of in valid voxels)
        dist = torch.clamp(dist / self.trunc_margin, min=-1) # here only sdf within 3 voxel distance are saved,

        # mask out voxels beyond trucaction distance behind surface
        valid1 = dist<1
        valid[valid.clone()] = valid[valid.clone()] * valid1
        dist = dist[valid1] # (n2) (where n2 is # of valid1 voxels)

        # where weight=0 copy in new values
        mask1 = self.weight_vol==0 # (nx*ny*nz)
        self.tsdf_vol[valid & mask1]  = dist[mask1[valid]]

        # where weight>0 and near surface, add in the value
        mask2 = valid.clone()
        valid2 = dist>-1
        mask2[valid] = mask2[valid] * valid2  # near surface
        mask3 = ~mask1 & mask2
        self.tsdf_vol[mask3]  += dist[mask3[valid]]
        self.weight_vol[mask2]+=1

        if self.color_vol is not None:
            self.color_vol[:, mask2] += color[:, py[mask2], px[mask2]]
        if self.label_vol is not None:
            self.label_vol[mask2] = label[py[mask2], px[mask2]] # newest label wins

    def get_tsdf(self, label_name='instance'):
        """ Package the TSDF volume into a TSDF data structure

        Args:
            label_name: name key to store label in TSDF.attribute_vols
                examples: 'instance', 'semseg'
        """

        nx, ny, nz = self.voxel_dim
        tsdf_vol = self.tsdf_vol.clone()
        tsdf_vol[self.weight_vol>0]/=self.weight_vol[self.weight_vol>0]
        tsdf_vol = tsdf_vol.view(nx,ny,nz)

        attribute_vols = {}
        if self.color_vol is not None:
            color_vol = self.color_vol.clone()
            color_vol[:, self.weight_vol>0]/=self.weight_vol[self.weight_vol>0]
            attribute_vols['color'] = color_vol.view(3,nx,ny,nz)
        if self.label_vol is not None:
            attribute_vols[label_name] = self.label_vol.view(nx,ny,nz).clone() # note it just init as -1 in __init__

        return TSDF(self.voxel_size, self.origin, tsdf_vol, attribute_vols)





