import multiprocessing as mp

import torch
import numpy as np
from tqdm import tqdm
from munch import Munch

from exp import ex
from ckpt import save_ckpt
from optimizer import get_optimizer
from utils import prepare_batch, get_all

from metrics.log import Logger, Metric
from metrics.wild360 import calc_score, visualize_heatmap, get_gt_heatmap


@ex.capture()
def train(log_path, config_dir, max_epoch, lr, clip_length, num_workers,
          display_config, num_frame, eval_start_epoch, save_model, model_name,
          use_exp_decay, use_epoch_decay, iterdecay_rate, epochdecay_rate):
    dataloaders, model = get_all()
    display_config = Munch(display_config)

    logger = Logger(log_path / config_dir)
    evaluator = Metric()

    # iters, for logging purpose
    it = 0
    val_it = 0

    optimizer, scheduler = get_optimizer(model=model,
                                         t_total=dataloaders['train'].dataset.t_total)
    optimizer.zero_grad()

    val_map_tracker_prev = {}
    val_map_tracker = {}

    for epoch in range(max_epoch):

        model.train()

        for batch in tqdm(dataloaders['train'], desc=f"Train{epoch:02d}"):
            if model_name in ['TokenCut', 'DINO', 'LOST']:
                break

            data, label, meta = prepare_batch(batch)
            data['frame_orig'] = data['frame']
            data['cls_orig'] = data['cls']
            label['mask_orig'] = label['mask']
            prev_frame = None

            for i in range(0, clip_length, num_frame):
                if prev_frame is not None:
                    label['prev_frame'] = prev_frame.cuda()

                if num_frame == 1:
                    data['frame'] = data['frame_orig'][:,i]
                    data['cls'] = data['cls_orig'][:,i]
                    label['mask'] = label['mask_orig'][:,i]
                elif num_frame > 1:
                    # Handling video input
                    data['frame'] = data['frame_orig'][:,i:i+num_frame]
                    data['cls'] = data['cls_orig'][:,i:i+num_frame]
                    label['mask'] = label['mask_orig'][:,i:i+num_frame]
                    if label['mask'].sum() == 0:
                        # no frames are valid 
                        # (mask indicates whether a frame is a zero-padded one or not)
                        break
                    elif label['mask'].sum() < num_frame:
                        # less than 5 frames are valid
                        # if feasible, frames from previous iteration are partially loaded
                        # if not, the input is discarded
                        diff = int(num_frame - label['mask'].sum())
                        if i < diff:
                            break
                        data['frame'] = data['frame_orig'][:,i-diff:i+num_frame-diff]
                        data['cls'] = data['cls_orig'][:,i-diff:i+num_frame-diff]
                        label['mask'] = label['mask_orig'][:,i-diff:i+num_frame-diff]
                else:
                    raise ValueError("Invalid num_frame value")

                result = model(data, label)

                if result['loss_total'] == 0.:
                    continue

                result['loss_total'].backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if num_frame == 1:
                    prev_frame = result['output'].detach()

                if it % 20 == 0:
                    logger.log_iter(it,
                                    optimizer.param_groups[0]['lr'],
                                    result,
                                    meta,
                                    'train')
                it += 1

                if use_exp_decay:
                    model.drop_prob *= iterdecay_rate

        if use_epoch_decay:
            model.drop_prob *= epochdecay_rate

        if epoch < eval_start_epoch:
            continue

        model.eval()

        val_split = 'val' if 'val' in dataloaders.keys() else 'test'
        val_dict = {}
        val_loss = []

        for batch in tqdm(dataloaders[val_split], desc=f"Val{epoch:02d}"):

            data, label, meta = prepare_batch(batch)
            data['frame_orig'] = data['frame']
            data['cls_orig'] = data['cls']
            label['mask_orig'] = label['mask']

            if num_frame == 1:
                result = model(data, label)
                if result['loss_total'] == 0.:
                    continue
                result['heatmap'] = model.compute_heatmap(result['output'].contiguous())
            elif num_frame > 1:
                # Processing video input
                # input is processed for every 5 frames
                # then outputs are concatenated altogether
                cnt = 0
                result = {'loss_total': 0, 'output': None}
                for i in range(0, clip_length, num_frame):
                    data['frame'] = data['frame_orig'][:,i:i+num_frame]
                    data['cls'] = data['cls_orig'][:,i:i+num_frame]
                    label['mask'] = label['mask_orig'][:,i:i+num_frame]
                    if label['mask'].sum() == 0:
                        break
                    elif label['mask'].sum() < num_frame:
                        if i == 0:
                            diff = 0
                            data['frame'] = data['frame_orig'][:,:int(label['mask'].sum())]
                            data['cls'] = data['cls'][:,:int(label['mask'].sum())]
                            label['mask'] = label['mask_orig'][:,:int(label['mask'].sum())]
                        else:
                            diff = int(num_frame - label['mask'].sum())
                            data['frame'] = data['frame_orig'][:,i-diff:i+num_frame-diff]
                            data['cls'] = data['cls_orig'][:,i-diff:i+num_frame-diff]
                            label['mask'] = label['mask_orig'][:,i-diff:i+num_frame-diff]
                    else:
                        diff = 0

                    result_temp = model(data, label)
                    cnt += 1
                    result['loss_total'] += result_temp['loss_total']
                    if result['output'] is None:
                        result['output'] = result_temp['output']

                        if model.verbose_mode:
                            result['output_cls'] = result_temp['output_cls']
                            result['output_time'] = result_temp['output_time']
                            result['output_space'] = result_temp['output_space']
                    else:
                        if diff > 0:
                            result_temp['output'] = result_temp['output'][:,diff:]

                            if model.verbose_mode:
                                result_temp['output_cls'] = result_temp['output_cls'][:, diff:]
                                result_temp['output_time'] = result_temp['output_time'][:, diff:]
                                result_temp['output_space'] = result_temp['output_space'][:, diff:]
                        result['output'] = torch.cat((result['output'], result_temp['output']), 1)

                        if model.verbose_mode:
                            result['output_cls'] = torch.cat((result['output_cls'], result_temp['output_cls']), 1)
                            result['output_time'] = torch.cat((result['output_time'], result_temp['output_time']), 1)
                            result['output_space'] = torch.cat((result['output_space'], result_temp['output_space']), 1)
                result['loss_total'] /= cnt

                result['heatmap'] = model.compute_heatmap(result['output'].contiguous())

                val_map_tracker[f'{meta["video_id"]}_{meta["segment_id"]}'] = result['heatmap']
                
                if model.verbose_mode:
                    result['heatmap_cls'] =  model.compute_heatmap(result['output_cls'].contiguous())
                    result['heatmap_time'] =  model.compute_heatmap(result['output_time'].contiguous())
                    result['heatmap_space'] =  model.compute_heatmap(result['output_space'].contiguous())
            else:
                raise ValueError(f"Invalid num_frame value: {num_frame}")

            data['frame'] = data['frame_orig']
            data['cls'] = data['cls_orig']
            label['mask'] = label['mask_orig']

            val_loss.append(result['loss_total'].item())

            for i in range(data['frame'].size()[0]):

                vid = meta['video_id'][i]
                sid = meta['segment_id'][i]
                no_frames = int(torch.sum(label['mask'][i]).item())   # To mark zero-padded frames
                if vid not in val_dict.keys():
                    val_dict[vid] = {'sal_map': [], 'gt_map': []}

                # Compute metrics per segment
                val_dict[vid]['sal_map'].extend([result['heatmap'][i][j].numpy() for j in range(no_frames)])
                # Naive gaussian smoothing
                # val_dict[vid]['sal_map'].extend([get_saliency_map(result['output'][i][j]) for j in range(no_frames)])
                val_dict[vid]['gt_map'].extend([label['gt'][i][j].cpu().numpy() for j in range(no_frames)])
                # Visualization
                if val_it % 30 == 0:
                    vis = {}

                    if display_config.overlay:
                        vis['video'] = torch.cat([visualize_heatmap(result['heatmap'][i][j],
                                                                    label['video'][i][j].cpu()).unsqueeze(0)
                                                  for j in range(no_frames)]).unsqueeze(0)
                    else:
                        vis['video'] = torch.cat([visualize_heatmap(result['heatmap'][i][j], overlay=False).unsqueeze(0)
                                                  for j in range(no_frames)]).unsqueeze(0)
                    vis['gt'] = torch.cat([get_gt_heatmap(label['gt'][i][j].cpu()).unsqueeze(0)
                                           for j in range(no_frames)]).unsqueeze(0)
                    # Naive gaussian smoothing with computational redundancy
                    # vis['video'] = torch.cat([overlay_heatmap(result['output'][i][j], use_gaussian=True, image=label['video'][i][j].cpu()).unsqueeze(0) for j in range(no_frames)]).unsqueeze(0)
                    # vis['gt'] = torch.cat([overlay_heatmap(label['gt'][i][j].cpu(), use_gaussian=False).unsqueeze(0) for j in range(no_frames)]).unsqueeze(0)
                    vis['orig'] = torch.from_numpy(label['video'][i][:no_frames].cpu().numpy().astype('uint8')).unsqueeze(0)

                    if display_config.concat:
                        cat_output = torch.cat([vis['video'], vis['gt'], vis['orig']], axis=-2)
                        vis = {}
                        vis['cat'] = cat_output

                        if model.verbose_mode:
                            salmap_cls = torch.cat([visualize_heatmap(result['heatmap_cls'][i][j], overlay=False).unsqueeze(0) for j in range(no_frames)]).unsqueeze(0)
                            salmap_time = torch.cat([visualize_heatmap(result['heatmap_time'][i][j], overlay=False).unsqueeze(0) for j in range(no_frames)]).unsqueeze(0)
                            salmap_space = torch.cat([visualize_heatmap(result['heatmap_space'][i][j], overlay=False).unsqueeze(0) for j in range(no_frames)]).unsqueeze(0)
                            salmap_cat = torch.cat([salmap_cls, salmap_time, salmap_space], axis=-2)
                            vis['cat'] = torch.cat([cat_output, salmap_cat], axis=-1)

                    logger.log_epoch(val_it, 0, vis, meta, 'eval')


                val_it += 1

            logger.log_epoch(epoch, 0, {'loss': sum(val_loss) / len(val_loss)}, meta, 'eval')

        if save_model:
            save_ckpt(epoch, 0, model)
        # Track four different metrics
        metrics = { m: [] for m in evaluator.get_metrics() }

        ## Collect score wrt videos
        for vid, scores in tqdm(val_dict.items(), desc=f"Score{epoch:02d}"):
            p = mp.Pool(num_workers)

            data_pairs = [(scores['sal_map'][i], scores['gt_map'][i]) for i in range(len(scores['sal_map']))]
            scores = [x for x in p.imap_unordered(calc_score, data_pairs)]

            p.close()
            p.join()

            for m in evaluator.get_metrics():
                metrics[m].append(sum([x[m] for x in scores]) / len(scores))

        metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
        evaluator.update(metrics)
        print(evaluator.get_result())
        logger.log_epoch(epoch, 0, metrics, meta, 'eval')

    print(evaluator.get_result())

    return evaluator.get_result()
