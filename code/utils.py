import wget
import math
import torch
import numpy as np
from torch import nn

from exp import ex
from ckpt import load_ckpt
from model import get_model
from data.dataloader import get_dataloaders


def download_url(url, out_path='.'):
    def pretty_size(val):
        if val // 1024 == 0:
            return f"{val}b"
        val = val // 1024
        if val // 1024 == 0:
            return f"{val}KB"
        val = val // 1024
        if val // 1024 == 0:
            return f"{val}MB"
        val = val // 1024
        return f"{val}GB"

    def bar_custom(current, total, width=80):
        width=30
        avail_dots = width-2
        shaded_dots = int(math.floor(float(current) / total * avail_dots))
        percent_bar = '[' + '■'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'
        progress = "%d%% %s [%s / %s]" % (current / total * 100, percent_bar, pretty_size(current), pretty_size(total))
        return progress

    wget.download(url, out=out_path, bar=bar_custom)

# convert data with types that cannot be processed with json.dumps() into str
def stringify_dict(raw_dict):
    new_dict = {}
    for k, v in raw_dict.items():
        if type(v) != dict:
            new_dict[k] = str(v)
        else:
            new_dict[k] = stringify_dict(v)
    return new_dict


@ex.capture()
def prepare_batch(batch, device):

    data, label, meta = batch

    for key, value in data.items():
        if isinstance(value, list):
            data[key] = [convert(v, device) for v in value]
        elif isinstance(value, dict):
            data[key] = {k: convert(v, device) for k, v in value.items()}
        else:
            data[key] = convert(value, device)

    for key, value in label.items():
        if isinstance(value, list):
            label[key] = [convert(v, device) for v in value]
        elif isinstance(value, dict):
            label[key] = {k: convert(v, device) for k, v in value.items()}
        else:
            label[key] = convert(value, device)

    return data, label, meta


def convert(value, device):
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if torch.is_tensor(value):
        value = value.to(device)
    return value


@ex.capture()
def get_all(device, modes=['train', 'test']):
    dataloaders = get_dataloaders(modes=modes)
    model = get_model()
    model = load_ckpt(model).to(device)

    return dataloaders, model
