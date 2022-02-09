# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from clu import metric_writers
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets import build_transform

import models_mae
import models_xlnet

from engine_pretrain import train_one_epoch, plot_evaluation_results

import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

import random
from PIL import Image, ImageFilter

class GaussianBlur(object):
    """Gaussian blur augmentation: https://github.com/facebookresearch/moco/"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_args_parser():
    parser = argparse.ArgumentParser('Pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--batch_size_val', default=8, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_ratio_range', default=None, type=float, nargs='+')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--span', default=[1], type=int, nargs='+')
    parser.add_argument('--betas', default=[0.9, 0.95], type=float, nargs='+')

    parser.add_argument('--clip_grad', default=None, type=float)

    parser.add_argument('--da', default='', type=str)

    parser.add_argument('--num_targets', default=None, type=int,
                        help='number of the visual tokens/patches to predict')
    parser.add_argument('--pred_pos_prob', default=0., type=float)
    parser.add_argument('--pred_pos_smoothing', default=0.1, type=float,
                        help='label smoothing for predicting position')
    parser.add_argument('--g_depth', default=0, type=int)
    # parser.add_argument('--alpha', default=0., type=float)
    parser.add_argument('--one_extra_layer', default=False, action='store_true')
    parser.add_argument('--avg_mask_token', default=False, action='store_true')
    parser.add_argument('--structured_ctx', default=False, action='store_true')
    parser.add_argument('--beit_ctx', default=False, action='store_true')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default=None, type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./logs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--tag', default=None, type=str)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def data_aug(da):
    if da == 'manual':
        tf_ = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif 'aa' in da:
        aa = da.replace("aa-", "")
        tf_ = transform = create_transform(
            input_size=args.input_size,
            scale=(0.2, 1.0),
            is_training=True,
            color_jitter=0,
            auto_augment=aa,
            interpolation='bicubic',
            re_prob=0,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
    else:
        tf_ = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return tf_

def main(args):
    misc.init_distributed_mode(args)

    if args.data_path is None:
        if os.path.isdir('/home/ubuntu/ILSVRC/Data/CLS-LOC/'):
            args.data_path = '/home/ubuntu/ILSVRC/Data/CLS-LOC/'
        elif os.path.isdir('/home/ubuntu/zhijie/ILSVRC/Data/CLS-LOC/'):
            args.data_path = '/home/ubuntu/zhijie/ILSVRC/Data/CLS-LOC/'
        elif os.path.isdir('/data/LargeData/Large/ImageNet/'):
            args.data_path = '/data/LargeData/Large/ImageNet/'
        else:
            exit(1)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    transform_train = data_aug(args.da)

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0:
        log_writer = metric_writers.create_default_writer(args.output_dir, asynchronous=False)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data_path, 'val'),
                             transform=build_transform(False, args)),
        sampler=None, shuffle=True,
        batch_size=args.batch_size_val,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    # define the model
    if 'mae' in args.model:
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    elif 'xlnet' in args.model:
        model = models_xlnet.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss,
            # pred_pos=args.pred_pos,
            # pred_pos_smoothing=args.pred_pos_smoothing,
            g_depth=args.g_depth,
            span=args.span,
            one_extra_layer=args.one_extra_layer,
            avg_mask_token=args.avg_mask_token,
            structured_ctx=args.structured_ctx,
            beit_ctx=args.beit_ctx)
        print("Num_seen", int(model.patch_embed.num_patches * (1 - args.mask_ratio)))

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(args.betas[0], args.betas[1]))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if 'xlnet' in args.model:
        plot_evaluation_results(model, data_loader_val, device, -1, log_writer, args)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if 'xlnet' in args.model:
            plot_evaluation_results(model, data_loader_val, device, epoch, log_writer, args)

        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.model == 'xlnet_vit_base_patch16':
        mo = 'xlnet_base_patch16_224'
    elif args.model == 'mae_vit_base_patch16':
        mo = 'mae_base_patch16_224'
    else:
        mo = args.model
    args.output_dir = os.path.join(args.output_dir, 'pretrain_' + mo)
    if args.tag is not None:
        args.output_dir += '_' + args.tag
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
