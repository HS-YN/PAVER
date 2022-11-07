import math

import cv2
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

from exp import ex


'''
3D rotation w.r.t. three axes
'''
def rotx(ang):
    return np.array([[1, 0, 0],
                     [0, np.cos(ang), -np.sin(ang)],
                     [0, np.sin(ang), np.cos(ang)]])


def roty(ang):
    return np.array([[np.cos(ang), 0, np.sin(ang)],
                     [0, 1, 0],
                     [-np.sin(ang), 0, np.cos(ang)]])


def rotz(ang):
    return np.array([[np.cos(ang), -np.sin(ang), 0],
                     [np.sin(ang), np.cos(ang), 0],
                     [0, 0, 1]])


'''
Conversion from one representation to another
'''
def sphere_to_twod(data, radius):
    # Input: [..., 2(theta, phi)]
    # Output: [..., 2(h, w)]
    C_sph = (0,0)    # theta, phi center coordinate

    out = np.zeros_like(data)
    out[..., 0] = radius * (data[..., 1] - C_sph[0])
    out[..., 1] = radius * (data[..., 0] - C_sph[1])
    return out


def sphere_to_threed(data):
    # Input: [..., 2(theta, phi)]
    # Output: [..., 3(x, y, z)]
    out = np.zeros_like(data)[..., :-1]
    out = np.concatenate([out, out, out], axis=-1)
    out[..., 0] = np.sin(data[..., 1]) * np.cos(data[..., 0])
    out[..., 1] = np.sin(data[..., 1]) * np.sin(data[..., 0])
    out[..., 2] = np.cos(data[..., 1])
    return out


def threed_to_sphere(data):
    # Input: [..., 3(x, y, z)]
    # Output: [..., 2(theta, phi)]
    out = np.zeros_like(data)[..., :-1]
    out[..., 0] = np.arctan2(data[..., 1], data[..., 0])
    z = np.sqrt(np.sum(data[..., :-1] * data[..., :-1], axis=-1))
    out[..., 1] = np.arctan2(z, data[..., 2])
    return out


def normalize_threed(data):
    # Normalize every 3D coordinate to norm=1
    out = np.sqrt(np.sum(data * data, axis=-1))
    for i in range(data.shape[-1]):
        data[..., i] /= out
    return data


def compute_patch(model_config, patch, ang_y, ang_z, is_discrete=False):
    '''
    Convert normal patch to deformed patch
    Input: 
        patch (tangential plane defined on (1,0,0))
        ang (angle to rotate vertically)
        (horizontal rotation is trivial)
        is_discrete (True if you need integer index to access pixel)
    Output:
        out (deformed patch)
    '''
    height = model_config['input_resolution']
    RES = (height, height * 2)
    RAD = RES[1] / (2 * np.pi)

    patch = patch @ roty(ang_y)
    patch = patch @ rotz(ang_z)
    out = normalize_threed(patch)
    out = threed_to_sphere(out)
    out = sphere_to_twod(out, RAD)

    # Handle overflow
    out[..., 0] = np.where(out[..., 0] > RES[0],
                           out[..., 0] - RES[0],
                           out[..., 0])
    out[..., 0] = np.where(out[..., 0] < 0,
                           out[..., 0] + RES[0],
                           out[..., 0])
    out[..., 1] = np.where(out[..., 1] > RES[1],
                           out[..., 1] - RES[1],
                           out[..., 1])
    out[..., 1] = np.where(out[..., 1] < 0,
                           out[..., 1] + RES[1],
                           out[..., 1])

    if is_discrete:
        out = out.astype(int)

    return out


def compute_all_patches(model_config, is_discrete=False):
    patch_size = model_config['patch_size']
    resolution = model_config['input_resolution']
    patch_no = resolution // patch_size  # 224 / 16 = 14

    P = np.arctan(np.pi / (patch_no * 2))
    R = patch_size

    # linspace for y is reverse in order, in order to make patch upright
    x, y = np.meshgrid(np.linspace(-P, P, R, dtype=np.float64),
                       np.linspace(P, -P, R, dtype=np.float64))
    x = np.expand_dims(x, -1)
    y = np.expand_dims(y, -1)
    z = np.ones_like(x)
    patch = np.concatenate([z, x, y], axis=-1)

    patches = []

    # To fix orientation issue, both for-loops iterate in a reverse order
    lat_range = patch_no // 2    # 7
    lon_range = patch_no * 2 - 1 # 27
    for lat in range(lat_range, -lat_range, -1):
        patches_lat = []
        for lon in range(lon_range, -1, -1):
            patch_lon = compute_patch(patch=patch,
                                      ang_y=P * (2 * lat - 1),
                                      ang_z=P * (2 * lon + 1),
                                      is_discrete=is_discrete,
                                      model_config=model_config)
            patches_lat.append(patch_lon)
        patches.append(patches_lat)

    return np.array(patches)


def compute_deform_offset(model_config, is_discrete=False):
    R = model_config['patch_size']

    patches = compute_all_patches(model_config=model_config, is_discrete=is_discrete)
    deform_offset = []

    for i in range(patches.shape[0]):
        col_offset = []

        for j in range(patches.shape[1]):
            # Destination (deformed patch)
            dst = patches[i, j].flatten()
            
            # Source (normal patch, before deformation)
            xx, yy = np.meshgrid(np.arange(R*j, R*(j+1)), np.arange(R*i, R*(i+1)))
            xx = np.expand_dims(xx, axis=-1)
            yy = np.expand_dims(yy, axis=-1)
            src = np.concatenate((yy, xx), axis=-1).flatten()
            
            col_offset.append(np.expand_dims(np.expand_dims(dst - src, axis=-1), axis=-1))
        # First concatenate w.r.t. last dimension (i.e., width)
        col_offset = np.concatenate(col_offset, axis=-1)

        deform_offset.append(col_offset)

    # Finally concatenate w.r.t. second last dimension (i.e., height)
    deform_offset = np.concatenate(deform_offset, axis=-2)

    # (16*16*2, 14, 28)
    return deform_offset


@ex.capture()
def visualize_patch(model_config, src):
    height = model_config['input_resolution']
    R = model_config['patch_size']

    dst = np.zeros_like(src)

    patches = compute_all_patches(is_discrete=True)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            idx = patches[i,j].astype(int).reshape((-1, 2)).T
            idx[0] %= height
            idx[1] %= height * 2

            dst[R*i:R*(i+1), R*j:R*(j+1)] = src[idx[0], idx[1]].reshape(R, R, 3)

    return dst