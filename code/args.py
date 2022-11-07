import re
import json
import random
import string
from datetime import datetime
from pathlib import Path, PosixPath

import torch
import jstyleson
import numpy as np
from munch import Munch
from sacred.arg_parser import get_config_updates

from config import args as default_args


def get_args(options, fixed_args={}, is_unittest=False):
    '''Processes arguments'''
    updated_args = {}
    updated_args.update(get_new_args(options))
    updated_args.update(fixed_args)

    root = Path('../').resolve()
    partial_configs = {}
    for k, v in default_args.items():
        if k.find('config_path') > -1:
            # Load json file with comments
            partial_configs[k] = jstyleson.load(open(str(root / Path(v))))

    for k, v in partial_configs.items():
        for kk, vv in v.items():
            if kk not in default_args.keys():
                default_args[kk] = vv
            elif type(vv) == dict:
                default_args[kk].update(vv)
            elif type(vv) == list:
                default_args[kk].extend(vv)
            else:
                print(f"(Warning: argument {kk} is being overridden. make sure that this is an expected behavior.)")
                default_args[kk] = vv
        del default_args[k]

    args = Munch(default_args)

    args = update_optional_args(args, updated_args)

    args.update(fix_seed(args))
    args.update(resolve_paths(args, is_unittest))

    if not args.debug:
        args.exp_id = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    else:
        args.exp_id = 'debug'

    args.config_dir = get_config_dir(args)
    args = args.toDict()

    # Primary assertions
    if args['device'] == 'cuda':
        assert torch.cuda.is_available(), "GPU device is not available"

    return args


def update_optional_args(args, updated_args):
    for k, v in updated_args.items():
        if type(v) != dict:
            args[k] = v
        else:
            if k not in args.keys():
                args[k] = {}
            args[k] = update_optional_args(args[k], v)
    return args


def get_new_args(options):
    '''Fetch updated arguments that deviate from default settings'''
    if 'UPDATE' in options:
        new_args, _ = get_config_updates(options['UPDATE'])
    else:
        new_args = options
    return new_args


def load_args(args, is_unittest=False):
    '''Load arguments of previous experiment'''
    if is_unittest:
        # Unit test is executed in `./code/unittest` directory
        root = Path('../../').resolve()
    else:
        root = Path('../').resolve()

    if str(root) not in str(args.ckpt_path):
        args.ckpt_path = root / args.ckpt_path
    if not hasattr(args, 'ckpt_name') or args.ckpt_name is None:
        return {}

    args_path = sorted(args.ckpt_path.glob(f'{args.ckpt_name}*'))
    if len(args_path) <= 0:
        return {}
    args_path = args_path[0] / 'args.json'
    ckpt_args = {}
    if args_path.is_file():
        ckpt_args = json.load(open(args_path, 'r'))['args']
        # update non-string arguments (and data_path)
        eval_keys = [k for k, v in default_args.items() if not isinstance(v, str)]
        eval_keys.append('data_path')
        # ckpt_args = {k: eval(v) if k in eval_keys else v for k, v in ckpt_args.items()}
        ckpt_args = {k: v for k, v in ckpt_args.items() if not k.endswith('_path')}
    return ckpt_args


def resolve_paths(args, is_unittest=False):
    '''Convert strings into paths if applicable'''
    path_list = [k for k in args.keys() if k.endswith('_path')] # and k != 'data_path']
    res_args = {}
    if is_unittest:
        # Unit test is executed in `./code/unittest` directory
        res_args['root'] = Path('../../').resolve()
    else:
        res_args['root'] = Path('../').resolve()
    for path in path_list:
        if args[path] is not None:
            if isinstance(args[path], list):
                res_args[path] = [res_args['root'] / Path(v) for v in args[path]]
            elif isinstance(args[path], dict):
                res_args[path] = {k: res_args['root'] / Path(v) for k, v in args[path].items()}
            else:
                res_args[path] = res_args['root'] / Path(args[path])
    return res_args


def fix_seed(args):
    '''Fix random seeds at once'''
    if 'random_seed' not in args or not isinstance(args['random_seed'], int):
        args['random_seed'] = args['seed'] if 'seed' in args else 0
    args['seed'] = args['random_seed'] # for sacred

    random.seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    torch.cuda.manual_seed_all(args['random_seed'])
    # torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    return args


def get_config_dir(args):
    '''Generate directory name for logging'''
    keys = [x for x in ['time', 'exp_id', 'tag'] if x in args.keys()]
    tags = [re.sub('[,\W-]+', '_', str(args[key])) for key in keys]
    dirname = '_'.join(tags)[:100] # Avoid too long paths
    return f"{dirname}"