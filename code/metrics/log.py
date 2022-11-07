import logging

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from exp import ex


eval_metric_dict = {
    'Wild360': {
       # 'metric_name': {'v': initial_value, 'f': priority function}
        'auc_borji':  {'v': 0, 'f': max},
        'auc_judd':   {'v': 0, 'f': max},
        'corr_coeff': {'v': 0, 'f': max},
        'similarity': {'v': 0, 'f': max},
    }
}


class Logger:
    def __init__(self, path):
        self.logger = SummaryWriter(path)
        self.cmd_logger = logging.getLogger(__name__)


    def __del__(self):
        # Finalizer to make sure all updates on summarywriter are properly reflected in log
        self.cmd_logger.info("Closing logger...")
        self.logger.flush()
        self.logger.close()


    def log_iter(self, it, lr, stat, meta, mode='train'):
        self.logger.add_scalar(f'{mode}/lr', lr, it)

        self._process_dict(it, stat, mode)


    def log_epoch(self, ep, lr, stat, meta, mode='eval'):
        self._process_dict(ep, stat, mode)


    def _process_dict(self, timestamp, stat, mode):
        for k, v in stat.items():
            if type(v) in [float, int, np.float64, np.float32, np.int8, np.uint8, np.int16]:
                self.logger.add_scalar(f'{mode}/{k}', v, timestamp)
            elif type(v) == torch.Tensor and v.dim() == 0:
                self.logger.add_scalar(f'{mode}/{k}', v.item(), timestamp)
            elif type(v) == str:
                self.logger.add_text(f'{mode}/{k}', v, timestamp)
            elif type(v) == torch.Tensor and v.dim() >= 3:
                if v.size()[-3] == 3:
                        if v.dim() >= 5:
                            self.logger.add_video(f'{mode}/{k}', v, timestamp, fps=10)
                        elif v.dim() >= 4:
                            self.logger.add_images(f'{mode}/{k}', v, timestamp)
                        else:
                            self.logger.add_image(f'{mode}/{k}', v, timestamp)


class Metric:
    @ex.capture()
    def __init__(self, dataset_name, priority=None):
        self.data = eval_metric_dict[dataset_name]
        for k in self.data.keys():
            self.data[k]['f'] = priority
        self.metrics = list(self.data.keys())


    def update(self, new):
        for k, v in self.data.items():
            if self.data[k]['f'] is not None:
                self.data[k]['v'] = self.data[k]['f'](self.data[k]['v'], new[k])
            else:
                # None -> FIFO
                self.data[k]['v'] = new[k]


    def get_metrics(self):
        return self.metrics


    def get_result(self):
        return {k:v['v'] for k, v, in self.data.items()}