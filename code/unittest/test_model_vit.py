import os
import sys
import wget
from unittest import TestCase

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.append('..')
from utils import download_url
from model.vit import VisionTransformer


class ViTModelSanity(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.url = "https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = VisionTransformer(patch_size=16,
                                       embed_dim=768,
                                       depth=12,
                                       num_classes=21843,
                                       num_heads=12,
                                       cls_only=False).to(self.device)
        self.model.eval()

        if not os.path.exists('assets/weight.npz'):
            os.makedirs('assets', exist_ok=True)
            download_url(self.url, out_path='assets/weight.npz')

        if not os.path.exists('assets/i21k_label.txt'):
            download_url("https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt", out_path='assets/i21k_label.txt')
        self.labels = open('assets/i21k_label.txt', 'r').read().splitlines()

    def test_output_patch_len(self):
        dummy_input = torch.randn((1, 3, 224, 224)).to(self.device)
        dummy_output = self.model(dummy_input)[0].squeeze()

        self.assertEqual(dummy_output.shape[0], 1 + (224 / 16) * (224 / 16))


    def test_model_load(self):
        # Visually inspect model shape
        print(self.model)

        dummy_input = torch.randn((1, 3, 224, 224)).to(self.device)
        dummy_output = self.model(dummy_input)[0].squeeze()

        self.model.load_pretrained(checkpoint_path='assets/weight.npz')

        dummy_output_p = self.model(dummy_input)[0].squeeze()

        self.assertNotEqual(torch.norm(dummy_output[0] - dummy_output_p[0]).item(), 0)


    def test_pretrained_validity(self):
        if not os.path.exists('assets/lion1.jpg'):
            os.makedirs('assets', exist_ok=True)
            download_url("https://www.krugerpark.co.za/images/black-maned-lion-shem-compion-590x390.jpg", out_path='assets/lion1.jpg')
            download_url("https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Lion_waiting_in_Namibia.jpg/220px-Lion_waiting_in_Namibia.jpg", out_path='assets/lion2.jpg')

        img1 = Image.open('assets/lion1.jpg').resize((224, 224))
        img2 = Image.open('assets/lion2.jpg').resize((224, 224))

        img1 = transforms.ToTensor()(img1).unsqueeze(0).to(self.device)
        img2 = transforms.ToTensor()(img2).unsqueeze(0).to(self.device)

        img1 = (img1 - 0.5) / 0.5
        img2 = (img2 - 0.5) / 0.5

        self.model.load_pretrained(checkpoint_path='assets/weight.npz')

        out1 = self.model(img1)[1].squeeze()
        out2 = self.model(img2)[1].squeeze()

        self.assertEqual(torch.argmax(out1).item(), torch.argmax(out2).item())
        self.assertEqual(self.labels[torch.argmax(out1).item()], "lion, king_of_beasts, Panthera_leo")


    def test_imagenet2012_weight(self):
        url_2 = "https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
        model_2 = VisionTransformer(patch_size=16,
                                    embed_dim=768,
                                    depth=12,
                                    num_classes=1000,
                                    num_heads=12,
                                    cls_only=False).to(self.device)
        model_2.eval()

        if not os.path.exists('assets/weight_2.npz'):
            download_url(url_2, out_path='assets/weight_2.npz')

        w = np.load('assets/weight_2.npz')
        prefix = ''
        if not prefix and 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'

        before = model_2.head.weight.detach().clone()
        model_2.load_pretrained(checkpoint_path='assets/weight_2.npz')
        after = model_2.head.weight

        self.assertNotEqual(torch.norm(before - after).item(), 0.)