import json

from exp import ex
from args import get_args

from utils import stringify_dict
from train import train as _train
from demo import demo as _demo

@ex.command
def train(_config):
    result = _train()

    return 0


@ex.command
def demo(_config):
    result = _demo()

    return 0

@ex.option_hook
def update_args(options):
    args = get_args(options)

    print(json.dumps(stringify_dict(args), indent=4))
    ex.add_config(args)
    return options


@ex.automain
def run():
    train()