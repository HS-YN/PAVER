import os
import sys
import wget
import json
from unittest import TestCase

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from munch import Munch

sys.path.append('..')
from exp import ex
from args import get_args
from utils import download_url

from geometry import *
from model.vit import ViT, DeformViT


class DeformPatchTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        @ex.automain
        def activate_sacred():
            ex.add_config(get_args({"debug": True}, is_unittest=True))
        ex.run()
        ex.run()


    @ex.capture()
    def get_model_variables(self, model_config, cache_path):
        return model_config, cache_path


    def test_patch_sampler(self):
        if not os.path.exists('assets/er.jpg'):
            download_url("https://wiki.panotools.org/images/2/20/Equirectangular.JPG", out_path='assets/er.jpg')
        src = cv2.imread('./assets/er.jpg')
        src = cv2.resize(src, (448,224))
        src = ((src / 255.) - 0.5) / 0.5
        dst = visualize_patch(src=src)

        model_config, cache_path = self.get_model_variables()
        model_vit = ViT(Munch(model_config), cache_path).cuda()
        model_vit.eval()

        model_deform = DeformViT(Munch(model_config), cache_path).cuda()
        model_deform.eval()

        src = torch.from_numpy(np.transpose(src, (2, 0, 1))).unsqueeze(0).float().cuda()
        dst = torch.from_numpy(np.transpose(dst, (2, 0, 1))).unsqueeze(0).float().cuda()

        src_out = model_deform(src).detach().cpu().squeeze()
        dst_out = model_vit(dst).detach().cpu().squeeze()

        self.assertTrue(torch.abs(src_out - dst_out).max() < torch.abs(src_out).max() * 0.0001)

