from vPlaneRecover.model_util import *

# The plane detection branch, where three semantic categories (floor, ceiling, wall) are handled together.
# The code is inspried by https://github.com/yanconglin/Deep-Hough-Transform-Line-Priors
# last modification: Fengting Yang 03/21/2022

def get_coords(voxel_dim):
    coords_x = np.arange(0, voxel_dim[0]) - voxel_dim[0]//2
    coords_y = np.arange(0, voxel_dim[1]) - voxel_dim[1]//2
    coords_z = np.arange(0, voxel_dim[2]) - voxel_dim[2]//2
    coords_x, coords_y, coords_z = np.meshgrid(coords_x, coords_y, coords_z, indexing='ij') # this is equal to default torch.meshgrid we used
    coords = np.stack((coords_x.flatten(), coords_y.flatten(), coords_z.flatten()))
    return coords


def hough_plane(voxel_dim, phi_step = 2, theta_step=90,  rho_step=2):
    # developed based on ht-lcnn HT.py
    # Phi and Theta ranges, where phi is yaw angle and theta is pitch angle
    phis = np.deg2rad(np.arange(0.0, 180.0, phi_step))
    thetas = np.deg2rad(np.arange(0.0, 91.0, theta_step))
    num_thetas, num_phis = len(thetas), len(phis)

    # Rho range
    # while in theory, we need the cube diag as rho, here we only take 0 and 90 deg theta,
    # so the max(hori_diag, vert_len) will be rho range
    row, col, hght = voxel_dim # [x//2 for x in voxel_dim]
    # diag = np.sqrt((row  - 1) ** 2 + (col  - 1) ** 2 + (hght -1)**2)
    diag = np.maximum(np.sqrt((row-1)**2+(col-1)**2), hght)
    q = np.ceil(diag / rho_step)
    num_rho = int(q + 1) #int(2 * q + 1)
    rhos = np.linspace(-q // 2 * rho_step, q // 2 * rho_step, num_rho)

    # Cache some resuable values
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    cos_p, sin_p = np.cos(phis),  np.sin(phis)

    sinT_cosP = sin_t[:, None] @ cos_p[None, :] # 12 * 6
    sinT_sinP = sin_t[:, None] @ sin_p[None, :]
    cosT = np.tile(cos_t[:, None], (1, num_phis))
    angle = np.stack([sinT_cosP, sinT_sinP, cosT]).reshape([3,-1])

    #get 3d coords
    coords = get_coords(voxel_dim)
    vote_map = (coords.T @ angle).astype(np.float32).reshape([-1, num_thetas, num_phis])  # x_sint_cosp + y_sint_cosp + z_cost

    vote_index = torch.zeros( num_rho, num_thetas, num_phis, int(row * col * hght),dtype=torch.uint8)

    for i in range(row * col * hght):
        for j in range(num_thetas):
            for k in range(num_phis):
                # only count theta == 0 once
                if thetas[j] == 0 and phis[k] != 0: continue
                rhoVal = vote_map[i, j, k]
                rhoIdx = np.nonzero(np.abs(rhos - rhoVal) == np.min(np.abs(rhos - rhoVal)))[0]
                vote_index[ rhoIdx[0], j, k, i] = 1

    vote_index = vote_index.reshape([-1, int(row * col * hght)]).to_sparse()

    return vote_index, rhos, thetas, phis, diag  # .reshape(rows, cols, h, w)


def make_conv_block(in_channels, out_channels, kernel_size=(3, 3,3), stride=1, padding=(1,1,1), dilation=1, groups=1, norm='BN',
    bias=False):
    # from HT-LCNN
    layers = []
    layers += [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]
    ###  no batchnorm layers
    layers += [get_norm_3d(norm, out_channels)]
    layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class Plane_HT(nn.Module):
    def __init__(self, channel, norm, drop):
        super(Plane_HT, self).__init__()

        self.n_map = 1  # len(SEM_INS_MAP)
        self.pre_convs =   conv1x1x1(channel, self.n_map * channel) #
        self.convs =  nn.ModuleList()
        self.proj = nn.ModuleList()


        for i in range(self.n_map):
            # self.pre_convs.append(conv1x1x1(channel, channel))
            self.convs.append( nn.Sequential(
                        make_conv_block(channel, channel, kernel_size=(3, 1, 3), stride=1, padding=(1, 0, 1),  norm=norm, bias=True),
                        make_conv_block(channel, channel, kernel_size=(3, 1, 3), stride=1, padding=(1, 0, 1),  norm=norm, bias=True),
                        make_conv_block(channel, channel, kernel_size=(3, 1, 3), stride=1, padding=(1, 0, 1),  norm=norm, bias=True),

                        make_conv_block(channel, channel, kernel_size=(3, 1, 3), stride=1, padding=(1, 0, 1),  norm=norm,bias=True),
                        make_conv_block(channel, channel, kernel_size=(3, 1, 3), stride=1, padding=(1, 0, 1),  norm=norm,bias=True),
                        make_conv_block(channel, channel, kernel_size=(3, 1, 3), stride=1, padding=(1, 0, 1),  norm=norm,bias=True)
            ))

            self.proj.append(conv1x1x1(channel, 1))

        self.top_k_ins = 10
        self.top_k_thres = 0.008 * self.n_map
        self.att_base = 1.

        # self.trans_norm1 = get_norm_3d(norm, channel)
        # self.trans_norm2 = get_norm_3d(norm, channel)

    def forward(self, feat_in, vote_idx, rhos, thetas, phis):
        b, c, w, d, h  = feat_in.shape

        sp, n_rho, n_tht, n_phi = vote_idx.shape[1], rhos.shape[0], thetas.shape[0], phis.shape[0]
        param_htmaps = []
        # att_maps = []
        _feat = self.pre_convs(feat_in)
        for i in range(self.n_map):
            # apply HT
            feat = _feat[:, i*c:(i+1)*c]
            feat = feat.reshape([b, c, -1]).reshape([int(b * c), -1])
            assert feat.shape[1] == vote_idx.shape[-1], \
                "feature shape must match vote_idx shape({}) for multiplication".format(vote_idx.shape)

            # # torch can only do sparse @ dense, normalize with diag square
            with torch.cuda.amp.autocast(enabled=False):
                HT_feat = (vote_idx @ feat.T.type(vote_idx.dtype)).T
                HT_feat = HT_feat.reshape([-1, n_rho, n_tht, n_phi]).reshape([b, c, n_rho, n_tht, n_phi])

                valid_cnt = torch.sparse.sum(vote_idx, dim=-1).to_dense().view(n_rho, n_tht, n_phi).unsqueeze(0).unsqueeze(0)
                # normalize with voxel cnt, same as projection normalization in inference2
                HT_feat = ((HT_feat) / (valid_cnt + 1e-4)).type(feat.dtype)# feat. is after relu always > 0 # / HT_feat.max()
                HT_feat[:, :, valid_cnt[0,0] == 0] = 0

            # HT_feat = self.trans_norm1(HT_feat)
            # param. space feat
            HT_feat = self.convs[i](HT_feat)

            # extract param. res
            param_htmap = self.proj[i](HT_feat).sigmoid()
            param_htmaps.append(param_htmap)

        ret_param_htmap = torch.cat(param_htmaps, dim=1)
        # generate attention map
        att_map = ret_param_htmap.sum(1)
        att_map = torch.where(att_map > self.top_k_thres, att_map, torch.zeros_like(att_map))
        att_map = att_map.reshape([b, -1])
        with torch.cuda.amp.autocast(enabled=False):
            HT_att = (vote_idx.transpose(1, 0) @ att_map.T.type(vote_idx.dtype)).T.type(feat.dtype)
            # IHT_feat = (vote_idx.transpose(1, 0).to_dense() @ HT_feat.T.type(vote_idx.dtype))
        HT_att = HT_att.reshape([b,1, w,d,h]).clamp(0, 1) + self.att_base

        return HT_att.detach(), ret_param_htmap



def load_vote_idx(voxel_dim, phi_step, theta_step, rho_step):
    vote_idx_pth = '../pln_proposal/vote_idx_{}_{}_{}_stp_{}_{}_{}.pth'.format(voxel_dim[0], voxel_dim[1], voxel_dim[2],
                                                               phi_step, theta_step, rho_step)
    if os.path.isfile(vote_idx_pth):
        # vote_idx = np.load(vote_idx_pth)
        vote_idx = torch.load(vote_idx_pth)
        thetas = np.deg2rad(np.arange(0.0, 91.0, theta_step))
        phis = np.deg2rad(np.arange(0.0, 180.0, phi_step))

        # Rho range
        # while in theory, we need the cube diag as rho, here we only take 0 and 90 deg theta,
        # so the max(hori_diag, vert_len) will be rho range
        row, col, hght = voxel_dim  # [x//2 for x in voxel_dim]
        # diag = np.sqrt((row - 1) ** 2 + (col - 1) ** 2 + (hght - 1) ** 2)
        diag = np.maximum(np.sqrt((row - 1) ** 2 + (col - 1) ** 2), hght)
        q = np.ceil(diag / rho_step)
        num_rho = int(q + 1)  # int(2 * q + 1)
        rhos = np.linspace(-q // 2 * rho_step, q // 2 * rho_step, num_rho)
    else:
        vote_idx, rhos, thetas, phis, diag = hough_plane(voxel_dim, phi_step, theta_step, rho_step)
        torch.save(vote_idx, vote_idx_pth)

    return  vote_idx, rhos, thetas, phis, diag
