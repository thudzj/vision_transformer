# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

from einops import rearrange
import numpy as np

import torchvision


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            if args.mask_ratio_range is None:
                mask_ratio = args.mask_ratio
            else:
                rng = np.random.RandomState(data_iter_step)
                mask_ratio = rng.uniform(args.mask_ratio_range[0], args.mask_ratio_range[1])
            # if data_iter_step < 20:
            #     print(data_iter_step, mask_ratio, int(model.module.patch_embed.num_patches * (1 - mask_ratio)), force=True)
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=mask_ratio, num_targets=args.num_targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    clip_grad=args.clip_grad,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            log_writer.write_scalars(
              len(data_loader) * epoch + data_iter_step,
              dict(
                  loss=loss_value_reduce,
                  lr=lr))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def plot_evaluation_results(model, data_loader_val, device, epoch, log_writer, args):
    mask_ratio = 0.99

    patch_size = model.module.patch_embed.patch_size
    grid_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])

    img = next(iter(data_loader_val))[0]
    img = img.to(device, non_blocking=True)

    _, outputs, target_indices = model(img, mask_ratio=mask_ratio)

    #save original img
    mean = torch.as_tensor([0.485, 0.456, 0.406]).to(device)[None, :, None, None]
    std = torch.as_tensor([0.229, 0.224, 0.225]).to(device)[None, :, None, None]
    ori_img = img * std + mean  # in [0, 1]

    img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[1])
    img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
    img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
    img_patch.scatter_(1, target_indices, outputs)

    #make mask
    mask = torch.ones_like(img_patch)
    mask.scatter_(1, target_indices, torch.zeros_like(outputs))
    mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
    mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
        p1=patch_size[0], p2=patch_size[1], h=grid_size[0], w=grid_size[1])

    #save reconstruction img
    rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
    rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) \
                + img_squeeze.mean(dim=-2, keepdim=True)

    rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
        p1=patch_size[0], p2=patch_size[1], h=grid_size[0], w=grid_size[1])

    #save random mask img
    img_mask = rec_img * mask

    if log_writer is not None:
        log_writer.write_images(
            epoch,
            dict(
                samples=ori_img.permute(0, 2, 3, 1).data.cpu().numpy(),
                vis=img_mask.permute(0, 2, 3, 1).data.cpu().numpy(),
                recon=rec_img.permute(0, 2, 3, 1).data.cpu().numpy(),)
            )
