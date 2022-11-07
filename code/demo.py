import multiprocessing as mp
import ffmpeg

import torch
import torchvision
import numpy as np
from tqdm import tqdm
from munch import Munch

from exp import ex
from ckpt import load_ckpt
from model import get_extractor
from optimizer import get_optimizer
from utils import prepare_batch, get_all

from metrics.log import Logger, Metric
from metrics.wild360 import calc_score, visualize_heatmap, get_gt_heatmap


@ex.capture()
def demo(log_path, config_dir, max_epoch, lr, clip_length, num_workers,
            display_config, num_frame, eval_start_epoch, save_model,
            input_video, model_config, input_format):
    # Extract feature from image
    model_config = Munch(model_config)

    probe = ffmpeg.probe(input_video)
    video_stream = next((stream for stream in probe['streams'] 
                         if stream['codec_type'] == 'video'), None)
    orig_width = int(video_stream['width'])
    orig_height = int(video_stream['height'])

    width = model_config.input_resolution * 2
    height = model_config.input_resolution

    cmd = (
        ffmpeg.input(input_video).filter('scale', width, height)
    )

    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
           .run(capture_stdout=True, quiet=True)
    )
    video = np.frombuffer(out, np.uint8)
    video = video.reshape([-1, height, width, 3])
    video = torch.from_numpy(video.astype('float32')).permute(0, 3, 1, 2)[5:10]
    video = ((video / 255.) - 0.5) / 0.5 # Google inception mean & std
    feature_extractor = get_extractor()

    feature_extractor = feature_extractor.cuda()    # self.model.cuda() throws error!
    feature_extractor.eval()

    encoded_feat = feature_extractor(video.cuda())

    iid = input_video.split('/')[-1][:-4]
    torch.save(encoded_feat, f'./qual/{iid}_feat.pt')

    del feature_extractor

    # Run inference
    dataloaders, model = get_all(modes=[])
    display_config = Munch(display_config)

    model = load_ckpt(model)

    model.eval()

    val_split = 'val' if 'val' in dataloaders.keys() else 'test'

    result = model({'frame': encoded_feat[:, 1:].unsqueeze(0), 'cls': encoded_feat[:, 0].unsqueeze(0)},
                   {'mask': torch.Tensor([1., 1., 1., 1., 1.]).unsqueeze(0).cuda()})
    result['heatmap'] = model.compute_heatmap(result['output'].contiguous())
    vis = torch.cat([visualize_heatmap(result['heatmap'][0][j], overlay=False).unsqueeze(0)
                     for j in range(5)]).unsqueeze(0)

    print(vis.size())

    torchvision.io.write_video(f'./qual/{iid}_out.mp4',
                               vis.squeeze(0).permute(0,2,3,1), fps=4)

    return 0