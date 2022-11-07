from datetime import datetime
from munch import Munch

args = {
    'tag': '',
    'debug': False,
    'num_workers': 32,
    'random_seed': 1234,
    'device': 'cuda',
    'time': datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),

    'extractor_config_path': './code/configs/extractor/deformvit.json',
    'data_config_path': './code/configs/dataset/wild360.json',
    'model_config_path': './code/configs/model/paver.json',

    'data_path': './data',
    'log_path': './data/log',                       # Overridden with local config
    'ckpt_path': './data/log',
    'cache_path': './data/cache',
    'rebuild_cache': False,
    'ckpt_name': None,

    'save_model': True,

    'optimizer_name': 'Adam',
    'scheduler_name': 'none',

    'display_config': {
        'concat': True, # Display gt, prop, video at once for brevity
        'overlay': False
    }
}