import math

import numpy as np
from torch.utils.data import DataLoader

from exp import ex
from .dataset import get_dataset


@ex.capture()
def get_dataloaders(batch_size, num_workers, max_epoch, clip_length, modes=['train', 'test']):
    dataset = get_dataset(modes=modes)
    outputs = {}

    for mode, ds in dataset.items():
        dataloader = DataLoader(ds,
                                batch_size=batch_size if mode == 'train' else 1,
                                # collate_fn=ds.collate_fn,
                                shuffle=(mode == 'train'),
                                num_workers=num_workers)
        # For gradient accumulation
        dataloader.dataset.t_total = math.ceil(len(ds) * max_epoch / batch_size) * clip_length
        outputs[mode] = dataloader
    return outputs