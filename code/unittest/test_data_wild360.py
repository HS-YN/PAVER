import os
import sys
import wget
import json
from pathlib import Path
from unittest import TestCase

import cv2
import torch

sys.path.append('..')
import data
from args import get_args
from exp import ex


class Wild360Dataset(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        @ex.automain
        def activate_sacred():
            ex.add_config(get_args({"rebuild_cache": False}, is_unittest=True))
        ex.run()


    @ex.capture()
    def get_data_path(self, _config):
        return _config['data_path']


    def test_dataset_existence(self):
        self.assertTrue('Wild360' in data.dataset_dict)


    def test_data_path_validity(self):
        data_path = self.get_data_path()
        self.assertTrue(data_path.exists())


    def test_dataset_size(self):
        dataset = data.dataset_dict['Wild360'](mode='train')
        vidlen = len(list(dataset.video_path.glob('*')))
        self.assertEqual(vidlen, 56)
        self.assertEqual(dataset.clips[0]['frame'].size()[0], 25)
        self.assertEqual(dataset.clips[0]['frame'].size()[1], 392)
        self.assertEqual(dataset.clips[0]['frame'].size()[2], 768)


    def test_shape_identity(self):
        dataset = data.dataset_dict['Wild360'](mode='test')
        self.assertEqual(len(list(dataset.video_path.glob('*'))), 29) # 24)
        for v in dataset.clips:
            k = v['video_id']
            width = v['width']
            height = v['height']
            video = v['frame'].size()
            gt = v['gt'].size()
            self.assertEqual(video[0], gt[0], f'Temporal inconsistency in {k}')