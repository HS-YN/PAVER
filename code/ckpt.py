import torch
from munch import Munch

from exp import ex


@ex.capture
def save_ckpt(epoch, loss, model, log_path, config_dir, _config):
    print(f'[LOG] Saving epoch {epoch:02d}')
    ckpt = {
        'args': _config,
        'epoch': epoch,
        'loss': loss,
        'model': model.state_dict()
    }

    ckpt_path = log_path / config_dir
    ckpt_path.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path / f"{epoch:02d}_ckpt.pt")


@ex.capture
def load_ckpt(model, ckpt_name, ckpt_path, model_config, cache_path):
    if ckpt_name is not None:
        name = f'{ckpt_name}*' if not ckpt_name.endswith('*') else f'{ckpt_name}'
        ckpt_path = sorted(ckpt_path.glob(name), reverse=False)
        assert len(ckpt_path) > 0, \
            "[ERROR] No checkpoint candidate for {}.".format(ckpt_name)
        ckpt_path = ckpt_path[0]
        print(f'[LOG] Loading checkpoint {ckpt_path}')
        data = torch.load(ckpt_path)
        if isinstance(data, dict) and 'model' in data:
            model.load_state_dict(data['model'])
        else:
            model.load_state_dict(data)
    return model