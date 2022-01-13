# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from timm.data import create_transform

from masking_generator import RandomMaskingGenerator
from dataset_folder import ImageFolder

from PIL import Image, ImageFilter
import random


class GaussianBlur(object):
    """Gaussian blur augmentation: https://github.com/facebookresearch/moco/"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class DataAugmentationForMAE(object):
    def __init__(self, args, for_val=False):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        if for_val:
            if args.input_size < 384:
                crop_pct = 224. / 256.
            else:
                crop_pct = 1.0
            size = int(args.input_size / crop_pct)
            self.transform = transforms.Compose([
                transforms.Resize(size, interpolation=3),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ])
        else:
            if 'xlnet' in args.model and args.pred_pos:
                jitter_d = 1.0
                jitter_p = 0.8
                grey_p = 0.2

                # blur_sigma = [0.1, 2.0]
                # blur_p = 0.5

                # guassian_blur from https://github.com/facebookresearch/moco/
                # guassian_blur = transforms.RandomApply([GaussianBlur(blur_sigma)], p=blur_p)

                color_jitter = transforms.ColorJitter(
                    0.8*jitter_d, 0.8*jitter_d, 0.8*jitter_d, 0.2*jitter_d)
                rnd_color_jitter = transforms.RandomApply([color_jitter], p=jitter_p)

                rnd_grey = transforms.RandomGrayscale(p=grey_p)

                self.transform = transforms.Compose([

                    transforms.RandomResizedCrop(args.input_size),
                    transforms.RandomHorizontalFlip(),
                    # guassian_blur,
                    rnd_color_jitter,
                    rnd_grey,
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
                ])

        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size, args.mask_ratio
        )

    def __call__(self, image):
        mask, target_indices = self.masked_position_generator()
        return self.transform(image), mask, target_indices

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))
    return ImageFolder(args.data_path, transform=transform)

def build_pretraining_val_dataset(args):
    transform = DataAugmentationForMAE(args, for_val=True)
    print("Data Aug = %s" % str(transform))
    return ImageFolder(args.data_path.replace('train', 'val'), transform=transform)

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
