import os
import logging
from pathlib import Path

from torch import nn
from munch import Munch
from inspect import getmro

from exp import ex


logger = logging.getLogger(__name__)

model_dict = {}


def full_model(target):
    # Decorator indicating full model
    target.is_full_model = True
    # target.run_pretrain = True

    return target

def add_models():
    path = Path(os.path.dirname(__file__))

    for p in path.glob('*.py'):
        name = p.stem
        parent = p.parent.stem
        if name != '__init__':
            __import__(f"{parent}.{name}")
            module = eval(name)
            for member in dir(module):
                member = getattr(module, member)
                if hasattr(member, '__mro__') and \
                        nn.Module in getmro(member) and \
                        hasattr(member, 'is_full_model'):
                    model_dict[str(member.__name__)] = member


def get_model_class(model):
    if not model_dict:
        add_models()

    assert model in model_dict.keys(), "[ERROR] Provided model \'{}\' does not exist. Possible candidates are: \n{}".format(model, str(model_dict.keys()))
    model = model_dict[model]
    return model


@ex.capture()
def get_model(model_name, model_config, cache_path):
    logger.info(f"Using model {model_name}")
    model = get_model_class(model_name)
    return model(Munch(model_config), cache_path)


@ex.capture()
def get_extractor(extractor_name, model_config, cache_path):
    logger.info(f"Using features extracted from {extractor_name}")
    model = get_model_class(extractor_name)
    return model(Munch(model_config), cache_path)