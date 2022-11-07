import logging
import multiprocessing as mp

import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from exp import ex


class FastLoader(Dataset):
    def __init__(self, data_dict):
        self.data_dict = [(k, v) for k, v in data_dict.items()]
        self.length = len(self.data_dict)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.data_dict[i]


def serial_save_by_segment(data, save_dir, no_seg=10):
    if type(data) == dict:
        key_list = list(data.keys())
        seg_size = len(key_list) // no_seg + 1

        data_list = []
        for i in range(no_seg):
            data_list.append({k: data[k] for k in key_list[i*seg_size:(i+1)*seg_size]})
    elif type(data) == list:
        seg_size = len(data) // no_seg + 1

        data_list = []
        for i in range(no_seg):
            data_list.append(data[i*seg_size:(i+1)*seg_size])
    else:
        assert False, "Data type not supported"


    for i in tqdm(range(len(data_list)), desc="Saving data"):
        torch.save(data_list[i], str(save_dir) + f'.part{i}')


def serial_load_by_segment(load_dir):
    data = {}
    file_list = list(load_dir.parent.glob(f'{load_dir.stem}*'))
    if type(torch.load(file_list[0])) == dict:
        data = {}
    else:
        data = []

    for file_i in tqdm(file_list):
        data_part = torch.load(file_i)

        if type(data_part) == dict:
            for data_item in data_part:
                for k, v in data_item.items():
                    data[k] = v
        elif type(data_part) == list:
            data.extend(data_part)
        else:
            assert False, "Data type not supported"
    return data


'''
Read dataset segments via multiprocessing by default
Empirically faster for larger files,
(>100G, 1.5x faster loading, 2.5x faster saving)
while difference is marginal for smaller ones
'''
def save_by_segment(data, save_dir, no_seg=10):
    if type(data) == dict:
        key_list = list(data.keys())
        seg_size = len(key_list) // no_seg + 1

        data_list = []
        for i in range(no_seg):
            data_list.append(({k: data[k] for k in key_list[i*seg_size:(i+1)*seg_size]}, str(save_dir) + f'.part{i}'))
    elif type(data) == list:
        seg_size = len(data) // no_seg + 1

        data_list = []
        for i in range(no_seg):
            data_list.append((data[i*seg_size:(i+1)*seg_size], str(save_dir) + f'.part{i}'))
    else:
        assert False, "Data type not supported"

    p = mp.Pool(5)

    for _ in tqdm(p.imap_unordered(save, data_list), total=len(data_list), desc="Saving data"):
        pass

    p.close()
    p.join()


def save(info):
    data = info[0]
    save_dir = info[1]
    torch.save(data, save_dir)


@ex.capture()
def load_by_segment(debug, load_dir):
    file_list = list(load_dir.parent.glob(f'{load_dir.stem}*'))
    

    if debug:
        logging.getLogger(__name__).warning(f'Loading small portion from the original dataset ({load_dir.stem}, debug=True)')
        return load(file_list[0])
        

    p = mp.Pool(5)

    raw_data = list(tqdm(p.imap_unordered(load, file_list), total=len(file_list), desc="Loading data"))

    p.close()
    p.join()

    if type(raw_data[0]) == dict:
        data = {}
        for data_part in raw_data:
            for k, v in data_part.items():
                data[k] = v
    elif type(raw_data[0]) == list:
        data = [x for sublist in raw_data for x in sublist]
    else:
        assert False, "Data type not supported"

    return data


def load(info):
    return torch.load(info)