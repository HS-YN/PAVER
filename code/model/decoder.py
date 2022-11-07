import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


from model.vit import Attention, Mlp, TempInterpDeformViT
from model.vit_utils import DropPath

from . import full_model, logger
from utils import download_url
from exp import ex
from metrics.wild360 import create_heatmap_with_spherical_gaussian_smoothing as basis_gen

from geometry import sphere_to_threed


class BaseDecoder(nn.Module):
    def __init__(self, model_config, cache_path):
        super().__init__()

        self.num_features = model_config.num_features
        self.num_classes = model_config.num_classes

        # Prepare adjacency matrix
        self.resolution = model_config.input_resolution
        self.patch_size = model_config.patch_size

        # Loss
        self.drop_prob = model_config.drop_prob

        # spherical gaussian smoothing for decoder
        sigma = model_config.sigma
        width = self.resolution * 2
        height = self.resolution
        patch_size = self.patch_size

        alpha = width * width / (4 * np.pi * np.pi * sigma * sigma)
        self.patch_cnt = height // patch_size  # 14

        self._create_adjacency_matrix((self.patch_cnt, 2 * self.patch_cnt), model_config.spatial_adjacency)

        basis_list = []
        if model_config.heatmap_geometry == "spherical":
            for j in range(self.patch_cnt):
                for i in range(self.patch_cnt * 2):
                    scale_factor = np.cos((2 * j + 1 - self.patch_cnt) * np.pi / (self.patch_cnt * 2))
                    coord = [[(2 * i + 1) * np.pi / (self.patch_cnt * 2),
                              (2 * j + 1 - self.patch_cnt) * np.pi / (self.patch_cnt * 2),
                              scale_factor * 1]]
                    basis_list.append(basis_gen(width, height, coord, alpha))
        elif model_config.heatmap_geometry == "cartesian":
            for j in range(self.patch_cnt):
                for i in range(self.patch_cnt * 2):
                    basis_map = np.zeros((height, width))
                    basis_map[j * self.patch_size + (self.patch_size // 2), i * self.patch_size + (self.patch_size // 2)] = 1.
                    basis_map = gaussian_filter(basis_map, sigma)
                    basis_list.append(basis_map)
        self.register_buffer('sph_basis', torch.from_numpy(np.array(basis_list)).float())

        spatial_weight = np.zeros((self.patch_cnt, self.patch_cnt * 2))
        for j in range(self.patch_cnt):
            for i in range(self.patch_cnt * 2):
                spatial_weight[j, i] = np.cos((2 * j + 1 - self.patch_cnt) * np.pi / (self.patch_cnt * 2))
        self.register_buffer('spatial_weight', torch.from_numpy(spatial_weight).flatten())


    def compute_heatmap(self, salmap):
        # salmap: model output (BTC)
        # BTC x CWH = BTWH
        heatmap = torch.einsum("...c,cwh->...wh", salmap ** 2, self.sph_basis)
        heatmap -= heatmap.min(-1)[0].min(-1)[0][:, :, None, None]    # BT11
        heatmap /= heatmap.max(-1)[0].max(-1)[0][:, :, None, None]    # BT11
        return heatmap.detach().cpu()


    def _create_adjacency_matrix(self, grid_size, geometry="cartesian"):
        if geometry == "cartesian":
            adjacency_matrix = []
            for i in range(grid_size[0] * grid_size[1]):
                matrix = np.zeros(grid_size)
                h_idx = i // grid_size[1]
                w_idx = i % grid_size[1]
                matrix[h_idx, (w_idx+1) % grid_size[1]] = 1.
                matrix[h_idx, w_idx-1] = 1.
                if h_idx > 0:
                    matrix[h_idx-1, w_idx] = 1.
                if h_idx < grid_size[0]-1:
                    matrix[h_idx+1, w_idx] = 1.
                matrix /= np.sum(matrix)
                adjacency_matrix.append(matrix.flatten())
            adjacency_matrix = torch.from_numpy(np.array(adjacency_matrix, dtype=np.float32))
            adjacency_matrix = torch.transpose(adjacency_matrix, 0, 1)

        elif geometry == "spherical":
            # Threshold: angular distance of single patch
            thres = sphere_to_threed(np.array([[0, 0], [0, np.pi / self.patch_cnt]]))
            thres = thres[0] @ thres[1].T

            xx, yy = np.meshgrid(np.arange(np.pi / (2 * self.patch_cnt), 2 * np.pi, np.pi / self.patch_cnt),
                                 np.arange(np.pi / (2 * self.patch_cnt), np.pi, np.pi / self.patch_cnt))
            ref_grid = np.concatenate((np.expand_dims(xx, axis=-1),
                                       np.expand_dims(yy, axis=-1)
                                      ), axis=-1).reshape(-1, 2)
            ref_grid = sphere_to_threed(ref_grid)

            # Map adjacency matrix from [0, 1] -> [thres, 1] -> [0, 1-thres]
            adjacency_matrix = ref_grid @ ref_grid.T - thres
            adjacency_matrix -= np.eye(adjacency_matrix.shape[0]) * (1 - thres)
            adjacency_matrix = np.where(adjacency_matrix > 0, adjacency_matrix, 0)
            adjacency_matrix /= adjacency_matrix.sum(1)
            adjacency_matrix = torch.from_numpy(adjacency_matrix).float()

        else:
            raise NotImplementedError(f"Provided geometry is not available for adjacency matrix: {geometry}")

        # Register adjacency matrix
        self.register_buffer('adjacency_matrix', adjacency_matrix)

    def forward(self, batch, label):
        raise NotImplementedError


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, C = x.shape
        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(B, T, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@full_model
class PAVER(BaseDecoder):
    def __init__(self, model_config, cache_path):
        super().__init__(model_config, cache_path)

        if model_config.cls_encoder_type == "mlp":
            self.cls_encoder = Mlp(in_features=model_config.num_features)
        elif model_config.cls_encoder_type == "identity":
            self.cls_encoder = nn.Identity()
        else:
            raise NotImplementedError
        self.attn_layer = Block(dim=model_config.num_features, num_heads=8)

        # Mfp related variables
        self.mask_prob = model_config.mask_prob

        self.lambda_cls = model_config.lambda_cls
        self.lambda_spatial_feat = model_config.lambda_spatial_feat
        self.lambda_temporal_feat = model_config.lambda_temporal_feat

        # Score related variables
        self.coeff_cls = model_config.coeff_cls
        self.coeff_time = model_config.coeff_time
        self.coeff_space = model_config.coeff_space

        # verbose mode for decoupling each score maps
        self.verbose_mode = model_config.verbose_mode

        # self.decoder_type = model_config.decoder_type
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.SmoothL1Loss(reduction='none')

        self.temp_score_weight = nn.Parameter(torch.ones(1))
        self.spat_score_weight = nn.Parameter(torch.ones(1))

    def forward(self, batch, label):
        features = batch['frame']
        if len(features.shape) == 3:
            features = features.unsqueeze(1) # BNC -> B1NC
            label['mask'] = label['mask'].unsqueeze(1)
            if label['mask'] < 0.1:
                return {'loss_total': 0.}
        
        predicted_features = self.attn_layer(features)
        
        result = {}
        result['loss_total'] = 0.

        cls_feat = self.cls_encoder(batch['cls'])
        cls_score = self.mse_loss(cls_feat.unsqueeze(-2), predicted_features).mean(-1)
        time_score = self.mse_loss(predicted_features.mean(-3).unsqueeze(-3), predicted_features).mean(-1)
        space_score = self.mse_loss(predicted_features.mean(-2).unsqueeze(-2), predicted_features).mean(-1)

        out = self.coeff_cls * cls_score + self.coeff_time * time_score + self.coeff_space * space_score

        result['cls_weight'] = self.mse_loss(cls_feat, batch['cls']).mean().item()

        if self.verbose_mode:
            result['output_cls'] = cls_score
            result['output_time'] = time_score
            result['output_space'] = space_score

        # cls loss
        cls_mean = cls_feat.detach() * label['mask'].unsqueeze(-1) # BTC
        cls_mean = cls_feat * (1 / label['mask'].sum(1).unsqueeze(-1).unsqueeze(-1)) # BTC
        loss_cls = self.mse_loss(cls_mean, cls_feat).mean(-1) # BT
        loss_cls = loss_cls.mean()
        result['loss_total'] += loss_cls * self.lambda_cls
        result['loss_cls'] = loss_cls.detach().item()

        # spatial feature consistency
        neighbor_avg = torch.einsum("BTNC,NN->BTNC",
                                    predicted_features,
                                    self.adjacency_matrix).detach()
        loss_feat_s = self.mse_loss(predicted_features - neighbor_avg,
                                    torch.zeros_like(predicted_features)).mean(-1) # BTN
        if 'mask' in label.keys():
            loss_feat_s = loss_feat_s.mean(-1) * label['mask']
        loss_feat_s = loss_feat_s.mean()
        result['loss_total'] += loss_feat_s * self.lambda_spatial_feat
        result['loss_feat_spatial'] = loss_feat_s.detach().item()

        # temporal feature consistency (BTNC)
        temporal_avg = predicted_features.detach().clone()
        temporal_avg[:,1:] = predicted_features.detach()[:,:-1]
        temporal_avg[:,0] = 0.
        temporal_avg[:,:-1] += predicted_features.detach()[:,1:]
        temporal_avg[:,1:-1] /= 2.

        loss_feat_t = self.mse_loss(predicted_features, temporal_avg.detach()).mean(-1) # BTN
        if 'mask' in label.keys():
            loss_feat_t = loss_feat_t.mean(-1) * label['mask']
        loss_feat_t = loss_feat_t.mean()
        result['loss_total'] += loss_feat_t * self.lambda_temporal_feat
        result['loss_feat_temporal'] = loss_feat_t.detach().item()

        result['output'] = out.detach()

        return result


@full_model
class FusePAVER(PAVER):
    # Space-time decoupled PAVER
    def __init__(self, model_config, cache_path):
        super().__init__(model_config, cache_path)
        self.attn_layer = FusedSTBlock(dim=model_config.num_features, num_heads=8)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


'''
Spatiotemporal Attention Block

qkv computes generic spatial attention of patches
while qkv2 computes temporal attention of patches using permutation
'''
class FusedSTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, C = x.shape
        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # QKV: BTHNC -> attn BTHNN -> x BTHNC -> x BTNHC -> BTNC
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(2, 3).reshape(B, T, N, C)

        # QKV BHNTC -> attn BHNTT -> x BHNTC -> BTNHC
        qkv2 = qkv.permute(0, 1, 3, 4, 2, 5) # 3BTHNC -> 3BHNTC
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = self.attn_drop(attn2.softmax(dim=-1))
        x2 = attn2 @ v2
        x2 = x2.permute(0, 3, 2, 1, 4)
        x2 = x2.reshape(B, T, N, C)

        x = self.proj((x + x2) * 0.5)
        x = self.proj_drop(x)
        return x


class FusedSTBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.attn = FusedSTAttention(dim=dim, num_heads=8)
