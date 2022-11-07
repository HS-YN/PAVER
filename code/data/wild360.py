import os
import json
import random
import logging
from pathlib import Path
from itertools import chain
import multiprocessing as mp

import cv2
import torch
import ffmpeg
import numpy as np
from tqdm import tqdm
from munch import Munch
from torch.utils.data import Dataset, DataLoader

from model import get_extractor
from exp import ex
from .utils import save_by_segment, load_by_segment, serial_save_by_segment, serial_load_by_segment, FastLoader

'''
Directory structure of Wild360 dataset
├train.txt
├test.txt
├Wild360_GT_29/{vid_id:13s}.mp4/{i:05d}.npy  # GT (960 x 1920)
└data/{train,test}/{vid_id:13s}.mp4
'''


class Wild360(Dataset):
    @ex.capture()
    def __init__(self, data_path, cache_path, rebuild_cache, num_workers,
                 clip_length, extractor_name, feature_norm, model_config,
                 eval_res, mode):
        super().__init__()
        model_config = Munch(model_config)

        self.name = 'wild360'
        self.split = mode
        self.logger = logging.getLogger(__name__)

        self.data_path = data_path
        self.cache_path = cache_path
        self.video_path = data_path / self.name / 'data' / self.split
        self.gt_path = data_path / self.name / 'Wild360_GT_29'

        self.rebuild_cache = rebuild_cache
        self.num_workers = num_workers
        self.input_resolution = model_config.input_resolution
        self.model_resolution = model_config.model_resolution
        self.patch_size = model_config.patch_size
        self.train_type = model_config.train_type
        self.clip_length = clip_length

        # Resolution for evaluation (applied to GT saliency map and video), following official implementation
        self.eval_res = (eval_res * 2, eval_res)  # (240, 120)

        # ViT Feature extractor
        self.extractor_name = extractor_name
        self.feature_norm = feature_norm

        # Load video (original length) and clip (equal-length cropping for training)
        with open(data_path / self.name / f'{self.split}.txt') as f:
            self.video_list = f.read().splitlines()
        self.clips = self.get_video()

    def __len__(self):
        return len(self.clips)


    def __getitem__(self, idx):
        clip = self.clips[idx]

        data = {
            'frame': clip['frame'],
            'cls': clip['cls'],
        }

        label = {
            # indicate padded frames incurred by equal-length temporal cropping
            'mask': torch.where(torch.norm(clip['frame'], dim=(-2, -1)) > 1e-6, 1., 0.)
        }
        if len(clip['frame'].size()) > 3:
            # Raw frame as input
            label['mask'] = torch.where(torch.norm(clip['frame'], dim=(-2, -1)).sum(-1) > 1e-5, 1., 0.)

        for key in ['gt', 'video']:
            if key in clip.keys():
                label[key] = clip[key]


        meta = {
            'width': clip['width'],
            'height': clip['height'],
            'video_id': clip['video_id'],
            'segment_id': clip['segment_id']
        }

        return data, label, meta


    def get_video(self):
        cache_file = f"{self.name}_{self.split}_f{self.clip_length}_"
        cache_file += f"r{self.input_resolution}_{self.extractor_name}_"
        cache_file += f"{self.model_resolution}_p{self.patch_size}_{self.train_type}.pkl"
        cache_file = self.cache_path / cache_file


        # Flush cache if necessary
        if self.rebuild_cache and len(list(cache_file.parent.glob(f'{cache_file.stem}*'))) > 0:
            for part_file in cache_file.parent.glob(f'{cache_file.stem}*'):
                part_file.unlink()

        # Load from cache if applicable
        if len(list(cache_file.parent.glob(f'{cache_file.stem}*'))) > 0:
            if self.input_resolution > 224 or self.patch_size == 8:
                clip_list = serial_load_by_segment(load_dir=cache_file)
            else:
                clip_list = load_by_segment(load_dir=cache_file)
        else:
            # Extract clips from video
            p = mp.Pool(self.num_workers)

            video_list = list(x for x in self.video_path.glob('*') if x.stem in self.video_list)
            video = {}
            for x in tqdm(p.imap_unordered(self._get_video, video_list),
                          total=len(video_list), desc='Wild360 VID'):
                video[x[0]] = {'frame': x[1], 'width': x[2], 'height': x[3]}

            p.close()
            p.join()

            if self.split == 'test':
                # Collect ground truth for test split
                gt_dirs = list(x for x in self.gt_path.glob('*') if x.stem in self.video_list)
                for gt_dir in tqdm(gt_dirs, desc='Wild360 GT'):
                    p = mp.Pool(self.num_workers)

                    video_gt_dict = {}
                    video_id = gt_dir.stem
                    
                    for x in p.imap_unordered(self._get_gt, gt_dir.glob('*.npy')):
                        video_gt_dict[x[0]] = x[1]
                    
                    video_gt = [video_gt_dict[i] for i in range(len(video_gt_dict))]
                    video_gt = np.array(video_gt).astype('float32')
                    video[video_id]['gt'] = torch.from_numpy(video_gt)

                    p.close()
                    p.join()

                    # Dimension sanity check
                    self.logger.debug(video_id, video[video_id]['width'], video[video_id]['height'])
                    self.logger.debug(video[video_id]['frame'].size(), video[video_id]['gt'].size())

            # split into item for batching
            clip_list = []

            # Prepare extractor model and dataloader
            loader = DataLoader(FastLoader(video))
            feature_extractor = get_extractor().cuda()
            feature_extractor.eval()

            for k, v in tqdm(loader, desc=f'Feature extraction ({self.extractor_name})'):
                v = {key: value[0] for key, value in v.items()} # remove batch dimension
                video_len = v['frame'].size()[0]

                for i in range(video_len // self.clip_length + 1):
                    if i * self.clip_length >= video_len:
                        continue
                    frame = v['frame'][i * self.clip_length: (i+1) * self.clip_length]

                    # Processing features with encoder
                    if self.feature_norm:
                        if self.train_type in ['dino']:
                            frame = frame / 255.
                            frame -= torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(1,3,1,1)).double()
                            frame /= torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape(1,3,1,1)).double()
                        else:
                            frame = ((frame / 255.) - 0.5) / 0.5
                    clip = {
                        'video_id': k,
                        'segment_id': i,
                        'frame': feature_extractor(frame.cuda()).detach().cpu(),
                        'width': v['width'],
                        'height': v['height']
                    }

                    if clip['frame'].size()[0] != self.clip_length:
                        # zero-pad for shorter clips
                        zero_vid = torch.zeros_like(clip['frame'][0])
                        if len(clip['frame'].size()) == 3:
                            zero_vid = zero_vid.repeat(self.clip_length - clip['frame'].size()[0], 1, 1)
                        else:
                            # Raw video input (len=4)
                            zero_vid = zero_vid.repeat(self.clip_length - clip['frame'].size()[0], 1, 1, 1)

                        clip['frame'] = torch.cat((clip['frame'], zero_vid), 0)

                    # Test split specific information
                    if self.split == 'test':
                        # Add original video for overlay
                        video = v['frame'][i * self.clip_length: (i+1) * self.clip_length].numpy()
                        video = [np.transpose(video[i], (1, 2, 0)) for i in range(video.shape[0])] # TCHW -> THWC
                        video = np.array([cv2.resize(x, self.eval_res, cv2.INTER_LANCZOS4) for x in video])
                        video = torch.from_numpy(np.transpose(video, (0, 3, 1, 2)))
                        if video.size()[0] != self.clip_length:
                            zero_vid = torch.zeros_like(video[0])
                            zero_vid = zero_vid.repeat(self.clip_length - video.size()[0], 1, 1, 1)
                            clip['video'] = torch.cat((video, zero_vid), 0)
                        else:
                            clip['video'] = video

                        # Add GT
                        gt = v['gt'][i * self.clip_length: (i+1) * self.clip_length]
                        if gt.size()[0] != self.clip_length:
                            # zero-pad for shorter clips
                            zero_vid = torch.zeros_like(gt[0])
                            zero_vid = zero_vid.repeat(self.clip_length - gt.size()[0], 1, 1)
                            clip['gt'] = torch.cat((gt, zero_vid), 0)
                        else:
                            clip['gt'] = gt


                    if len(clip['frame'].size()) == 3:
                        clip['cls'] = clip['frame'][:, 0]    # T(N+1)C -> TC
                        clip['frame'] = clip['frame'][:, 1:] # T(N+1)C -> TNC
                    else:
                        # Placeholder
                        clip['cls'] = clip['frame'][:, 0]

                    clip_list.append(clip)
            
            # Save data
            if self.input_resolution > 224 or self.patch_size == 8:
                # Due to bus error (core dumped), lack of shared memory
                serial_save_by_segment(data=clip_list, save_dir=cache_file)
            else:
                save_by_segment(data=clip_list, save_dir=cache_file)

        return clip_list


    def _get_video(self, vid):
        '''
        This function is adapted from Miech's video feature extractor with modification
        https://github.com/antoine77340/video_feature_extractor/blob/master/video_loader.py
        '''
        video_id = vid.stem
        is_panorama = True

        # Compute original resolution (for mainly eval purpose)
        probe = ffmpeg.probe(vid)
        video_stream = next((stream for stream in probe['streams'] 
                             if stream['codec_type'] == 'video'), None)
        orig_width = int(video_stream['width'])
        orig_height = int(video_stream['height'])

        if is_panorama:
            width = self.input_resolution * 2
            height = self.input_resolution
        else:
            width = orig_width
            height = orig_height
            width = self.input_resolution if width < height else self.input_resolution * width // height
            height = self.input_resolution if width >= height else self.input_resolution * height // width

        cmd = (
            ffmpeg.input(vid)#.filter('fps', fps=self.fps)
                             .filter('scale', width, height)
        )

        if not is_panorama:
            x = (width - self.input_resolution) // 2
            y = (height - self.input_resolution) // 2
            cmd = cmd.crop(x, y, self.input_resolution, self.input_resolution)

        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
               .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8)
        video = video.reshape([-1, height, width, 3])
        video = torch.from_numpy(video.astype('float32')).permute(0, 3, 1, 2)
        
        return video_id, video, orig_width, orig_height


    def _get_gt(self, gt):
        overlay_id = int(gt.stem)
        return overlay_id, cv2.resize(np.load(gt), self.eval_res, cv2.INTER_LANCZOS4)