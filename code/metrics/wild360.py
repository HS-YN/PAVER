import cv2
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal

from exp import ex

EPS = 1e-13

# Multiprocessing is necessary due to low speed
def calc_score(pair):
    sal_map = pair[0]
    gt_map = pair[1]

    return {
        'auc_borji': AUC_Borji(sal_map, gt_map),
        'auc_judd': AUC_Judd(sal_map, gt_map),
        'corr_coeff': CorrCoeff(sal_map, gt_map),
        'similarity': similarity(sal_map, gt_map)
    }


def AUC_Borji(saliency_map, fixation_map, base_shape=(240, 120), Nsplits=100, stepSize=0.01, save_fig=None):

    score = float('nan')

    # resize maps to base_shape
    saliency_map = cv2.resize(saliency_map, base_shape, cv2.INTER_LANCZOS4)
    fixation_map = cv2.resize(fixation_map, base_shape, cv2.INTER_LANCZOS4)
    assert saliency_map.shape == fixation_map.shape

    # if jitter:
    #    # jitter the saliency map slightly to disrupt ties of same numbers
    #    sshape = saliency_map.shape
    #    saliency_map = saliency_map+np.random.randn(sshape[0], sshape[1])/1e7

    # normalize saliency map
    saliency_map[saliency_map > np.mean(saliency_map) + 2 * np.std(saliency_map)] = 1.0
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))

    if np.sum(np.isnan(saliency_map)) == np.size(saliency_map):
        print('NaN saliency_map')
        exit()

    S = saliency_map.flatten()
    F = fixation_map.flatten()

    Sth = S[F > np.mean(F) + 2 * np.std(F)]  # sal map values at fixation locations
    Nfixations = np.size(Sth)
    Npixels = np.size(S)

    rr = np.random.randint(0, high=Npixels, size=(Nfixations, Nsplits))
    randfix = S[rr]

    auc = []

    for ss in range(Nsplits):
        curfix = randfix[:, ss]
        try:
            allthreshes = np.arange(0.0, np.max(np.append(Sth, curfix)), stepSize)[::-1]
        except:
            print("allthreshes wrong")
        # allthreshes = np.sort(Sth)[::-1]    # descend
        tp = np.zeros(len(allthreshes)+2)
        fp = np.zeros(len(allthreshes)+2)
        tp[0] = 0.0
        tp[-1] = 1.0
        fp[0] = 0.0
        fp[-1] = 1.0

        for i, thresh in enumerate(allthreshes):
            #aboveth = np.sum(S >= thresh)
            tp[i+1] = np.sum(Sth >= thresh)/float(Nfixations)
            fp[i+1] = np.sum(curfix >= thresh)/float(Nfixations)

        auc.append(np.trapz(tp, fp))

        #allthreshes = np.concatenate(([1], allthreshes, [0]))
    score = np.mean(auc)
    if save_fig is not None:
        plt.plot(fp, tp, 'b-')
        plt.title('Area under ROC curve: {}'.format(score))
        plt.savefig(save_fig)

    return score


def AUC_Judd(saliency_map, fixation_map, base_shape=(240, 120), jitter=True, save_fig=None):
    
    score = float('nan')

    saliency_map = cv2.resize(saliency_map, base_shape, cv2.INTER_LANCZOS4)
    fixation_map = cv2.resize(fixation_map, base_shape, cv2.INTER_LANCZOS4)
    assert saliency_map.shape == fixation_map.shape

    if jitter:
        # jitter the saliency map slightly to disrupt ties of same numbers
        sshape = saliency_map.shape
        saliency_map = saliency_map + np.random.randn(sshape[0], sshape[1]) / 1e7

    # normalize saliency map
    #saliency_map[saliency_map > np.mean(saliency_map)+2*np.std(saliency_map)] = 1.0
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))

    if np.sum(np.isnan(saliency_map)) == np.size(saliency_map):
        print('NaN saliency_map')
        exit()

    S = saliency_map
    F = fixation_map

    Sth = S[F > np.mean(F) + 2 * np.std(F)]  # sal map values at fixation locations
    Nfixations = np.size(Sth)
    Npixels = np.size(S)

    allthreshes = np.sort(Sth)[::-1]    # descend
    tp = np.zeros(Nfixations+2)
    fp = np.zeros(Nfixations+2)
    tp[0] = 0.0
    tp[-1] = 1.0
    fp[0] = 0.0
    fp[-1] = 1.0

    for i, thresh in enumerate(allthreshes):
        aboveth = np.sum(S >= thresh)
        tp[i+1] = i/Nfixations
        fp[i+1] = (aboveth-i)/(Npixels-Nfixations)

    score = np.trapz(tp, fp)
    allthreshes = np.concatenate(([1], allthreshes, [0]))

    if save_fig is not None:
        plt.plot(fp, tp, 'b-')
        plt.title('Area under ROC curve: {}'.format(score))
        plt.savefig(save_fig)
        
    return score


def CorrCoeff(map1, map2, base_shape=(240, 120)):
    map1 = cv2.resize(map1, base_shape, cv2.INTER_LANCZOS4)
    map2 = cv2.resize(map2, base_shape, cv2.INTER_LANCZOS4)
    assert map1.shape == map2.shape

    map1 = (map1-np.mean(map1))/np.std(map1)
    map2 = (map2-np.mean(map2))/np.std(map2)

    k = np.shape(map1)
    H = k[0]
    W = k[1]
    c = np.zeros((H, W))
    d = np.zeros((H, W))
    e = np.zeros((H, W))

    # Calculating mean values
    AM = np.mean(map1)
    BM = np.mean(map2)
    # Vectorized versions of c,d,e
    c_vect = (map1-AM)*(map2-BM)
    d_vect = (map1-AM)**2
    e_vect = (map2-BM)**2

    # Finally get r using those vectorized versions
    r_out = np.sum(c_vect)/float(np.sqrt(np.sum(d_vect)*np.sum(e_vect)))

    return r_out


def similarity(map1, map2, base_shape=(240, 120)):
    map1 = cv2.resize(map1, base_shape, cv2.INTER_LANCZOS4)
    map2 = cv2.resize(map2, base_shape, cv2.INTER_LANCZOS4)
    assert map1.shape == map2.shape

    map1 = (map1-np.min(map1))/(np.max(map1)-np.min(map1))
    map1 = map1/np.sum(map1)
    map2 = (map2-np.min(map2))/(np.max(map2)-np.min(map2))
    map2 = map2/np.sum(map2)
    score = np.sum(np.minimum(map1, map2))
    return score


def create_heatmap_with_spherical_gaussian_smoothing(width, height, coordinates, a = 5.0):
    heatmap = np.zeros((height, width))

    coef = a / np.sinh(a)    
    for coordinate in coordinates:
        theta, phi, value = coordinate
        
        phi_range = np.linspace(-np.pi / 2., np.pi / 2., height, endpoint=False)
        theta_range = np.linspace(0., 2 * np.pi, width, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, sparse=False)
        
        psi = 2 * np.arcsin(np.sqrt(np.square(np.sin((phi_grid - phi) / 2.0)) + np.cos(phi_grid) * np.cos(phi) * np.square(np.sin((theta_grid - theta) / 2.0))))
        heatmap += value * coef * np.exp(a * np.cos(psi))
        
    return heatmap


def visualize_heatmap(heatmap, image=None, overlay=True):
    fg = Image.fromarray(plt.get_cmap('jet')(heatmap.numpy(), bytes=True)[:,:,:3], mode='RGB').resize((240, 120))
    if not overlay:
        return torch.from_numpy(np.transpose(np.asarray(fg), (2, 0, 1)))
    else:
        bg = Image.fromarray(np.transpose(image.numpy().astype('uint8'), (1, 2, 0)))

        result = np.asarray(Image.blend(bg, fg, 0.5))
        result = np.transpose(result, (2, 0, 1))
        
        return torch.from_numpy(result)


def get_gt_heatmap(heatmap):
    h_map = heatmap.numpy()
    h_map = h_map - np.min(h_map)
    h_map = h_map / np.max(h_map)
    h_map = plt.get_cmap('jet')(h_map, bytes=True)
    h_map = np.transpose(h_map[...,:3], (2, 0, 1))

    return torch.from_numpy(h_map)


if __name__ == '__main__':
    # generate points 
    width, height = 448, 224
    coordinates = [
        [2 * np.pi / 8, np.pi / 4, 0],#.8],
        [29 * np.pi / 28, -np.pi / 28, 1.0],
        [7 * 2 * np.pi / 8, -np.pi / 4, 0]#.2]
    ]

    heatmap = create_heatmap_with_spherical_gaussian_smoothing(width, height, coordinates, 12)
    plt.imshow(heatmap)
    plt.savefig('./test_heatmap.jpg')
    
    roll_heatmap = np.roll(heatmap, heatmap.shape[1] // 2, axis=1)
    plt.imshow(roll_heatmap)
    plt.savefig('./test_heatmap_roll.jpg')