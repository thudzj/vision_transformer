# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import numpy as np

import torch
import torch.nn as nn

import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

class MyCrossEntropyLoss(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, input, target):
        logp = input.log_softmax(-1)
        return -torch.mean(torch.sum(logp * target, dim=-1))

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler,
                    max_norm: float = 0, window_size = (14, 14), patch_size: int = 16,
                    normlize_target: bool = True, num_targets: int = None,
                    pred_pos: bool = False, pred_pos_smoothing: float = 0.,
                    log_writer=None, hooks=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    loss_func = nn.MSELoss() if not pred_pos else MyCrossEntropyLoss()

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos, target_indices = batch
        images = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        target_indices = target_indices.to(device, non_blocking=True).flatten(1).to(torch.long)

        # import pdb; pdb.set_trace()
        with torch.no_grad():

            if num_targets is not None:
                target_indices = target_indices[:, :num_targets]

            if not pred_pos:
                # calculate the predict label
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
                unnorm_images = images * std + mean  # in [0, 1]

                if normlize_target:
                    images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                    images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                        ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                    # we find that the mean is about 0.48 and standard deviation is about 0.08.
                    images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
                else:
                    images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

                labels = torch.take_along_dim(images_patch, target_indices[:, :, None], dim=1)
            else:
                row_labels = torch.div(target_indices.reshape(-1, 1, 1), window_size[0], rounding_mode='trunc')
                col_labels = target_indices.reshape(-1, 1, 1) % window_size[0]
                new_labels = torch.exp(-((row_labels - torch.arange(window_size[0], device=device).view(1, -1, 1)) ** 2 +
                    (col_labels - torch.arange(window_size[0], device=device).view(1, 1, -1)) ** 2) / 2. / (pred_pos_smoothing + 1e-8))
                labels = new_labels.view(target_indices.shape[0], target_indices.shape[1], -1)
                labels = labels / torch.sum(labels, dim=-1, keepdim=True)

        with torch.cuda.amp.autocast():
            outputs = model(images, bool_masked_pos, target_indices)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None and step % print_freq == 0:
            log_writer.write_scalars(
              start_steps + step,
              dict(
                  loss=loss_value,
                  loss_scale=loss_scale_value,
                  lr=max_lr,
                  min_lr=min_lr,
                  weight_decay=weight_decay_value,
                  grad_norm=grad_norm.item()))

        if hooks is not None:
            for hook in hooks:
                hook(step)

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def plot_evaluation_results(model, data_loader_val, device, epoch, pred_pos,
                            patch_size, window_size, log_writer):
    batch = next(iter(data_loader_val))[0]

    img, bool_masked_pos, target_indices = batch
    img = img.to(device, non_blocking=True)
    bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
    target_indices = target_indices.to(device, non_blocking=True).flatten(1).to(torch.long)

    outputs = model(img, bool_masked_pos, target_indices)
    if pred_pos:
        recon = outputs.argmax(-1)
        recon_ = []
        for idx_b in range(recon.shape[0]):
          recon_b = []
          for idx_p in range(recon.shape[1]):
            recon_b.append(utils.pos2img(recon[idx_b, idx_p],
                                         window_size[1],
                                         patch_size[0], patch_size[1]))
          recon_.append(np.stack(recon_b, 0).reshape(recon.shape[1], -1))
        outputs = torch.from_numpy(np.stack(recon_, 0)).to(device).type_as(img)

    #save original img
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
    ori_img = img * std + mean  # in [0, 1]

    img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[1])
    img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
    img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
    for idx_b in range(img_patch.shape[0]):
        img_patch[idx_b][bool_masked_pos[idx_b]] = outputs[idx_b]

    #make mask
    mask = torch.ones_like(img_patch)
    for idx_b in range(mask.shape[0]):
        mask[idx_b][bool_masked_pos[idx_b]] = 0
    mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
    mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
        p1=patch_size[0], p2=patch_size[1], h=window_size[0], w=window_size[1])

    #save reconstruction img
    rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
    if not pred_pos:
        rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) \
                    + img_squeeze.mean(dim=-2, keepdim=True)
    rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
        p1=patch_size[0], p2=patch_size[1], h=window_size[0], w=window_size[1])

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
