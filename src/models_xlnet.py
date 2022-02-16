# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torchvision

from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.layers import DropPath, trunc_normal_
from util.pos_embed import get_2d_sincos_pos_embed

import math
import sys
import numpy as np

import random
import math
import numpy as np

def beit_mask(height, width, num_masks, min_num_patches=16, max_num_patches=None,
              min_aspect=0.3, max_aspect=None):

    min_num_patches = min(min_num_patches, num_masks)
    max_num_patches = num_masks if max_num_patches is None else max_num_patches
    max_aspect = max_aspect or 1 / min_aspect
    def _mask(mask, max_mask_patches):
        delta = 0
        while 1:
            target_area = random.uniform(min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(math.log(min_aspect), math.log(max_aspect)))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < width and h < height:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    mask = torch.zeros(height, width, dtype=int)
    mask_count = 0
    while mask_count < num_masks:
        delta = _mask(mask, num_masks - mask_count)
        if delta == 0:
            break
        else:
            mask_count += delta

    ids_ctx = mask.view(-1).nonzero().view(-1)
    others = (1 - mask.view(-1)).nonzero().view(-1)
    others = others[torch.randperm(others.shape[0])].view(others.size())
    ids_shuffle = torch.cat([ids_ctx, others])
    return ids_shuffle


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x2=None, select_kv=None, attn_mask=None):
        B, N, C = x.shape
        if x2 is None:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        else:
            B2, N2, _ = x2.shape
            bias1 = None if self.qkv.bias is None else self.qkv.bias[:self.qkv.bias.shape[0]//3]
            bias2 = None if self.qkv.bias is None else self.qkv.bias[self.qkv.bias.shape[0]//3:]
            q = nn.functional.linear(x, self.qkv.weight[:self.qkv.weight.shape[0]//3], bias1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = nn.functional.linear(x2, self.qkv.weight[self.qkv.weight.shape[0]//3:], bias2).reshape(B2, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        if select_kv is not None:
            k = k[:, :, :select_kv, :]
            v = v[:, :, :select_kv, :]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn_mask = attn_mask.float().masked_fill(~attn_mask, float('-inf')).masked_fill(attn_mask, float(0.0))
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x2=None, select_kv=None, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), x2 if x2 is None else self.norm1(x2), select_kv, attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x




class XLNetViT(nn.Module):
    """ XLNet with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 norm_pix_loss=False,
                 g_depth=0, span=[1], one_extra_layer=False, avg_mask_token=False,
                 structured_ctx=False, beit_ctx=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])

        if one_extra_layer:
            self.extra_block = Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)

        if g_depth > 0:
            self.g_blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                for i in range(g_depth)])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, patch_size**2 * in_chans, bias=True)

        # self.norm_2 = norm_layer(embed_dim)
        # self.head_2 = nn.Linear(embed_dim, 1000, bias=True)

        # --------------------------------------------------------------------------

        self.span = span
        self.norm_pix_loss = norm_pix_loss
        self.g_depth = g_depth
        self.one_extra_layer = one_extra_layer
        self.avg_mask_token = avg_mask_token
        self.structured_ctx = structured_ctx
        self.beit_ctx = beit_ctx

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_encoder(self, x, num_seen, num_targets, attn_mask):
        # embed patches
        x_ = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x_ + self.pos_embed[:, 1:, :]
        if self.avg_mask_token:
            g = self.pos_embed[:, 1:, :] + x_.mean(1)[:, None, :]
        else:
            g = (self.pos_embed[:, 1:, :] + self.mask_token).expand_as(x_)

        # permutation auto-regressive modeling
        N, L, D = x.shape  # batch, length, dim

        if self.beit_ctx:
            tmp = int(math.sqrt(self.patch_embed.num_patches))
            ids_shuffle = torch.stack([beit_mask(tmp, tmp, num_seen) for _ in range(N)]).to(x.device)
        elif self.structured_ctx:
            tmp = int(math.sqrt(self.patch_embed.num_patches))
            hs = torch.randint(max(1, int(math.ceil(float(num_seen) / tmp))), min(tmp, num_seen) + 1, (N,))
            ws = (float(num_seen) / hs).ceil().long()

            ws_start = ((tmp - ws + 1) * torch.rand(N)).int()
            hs_start = ((tmp - hs + 1) * torch.rand(N)).int()

            tmpm = torch.arange(0, tmp)[:, None] * tmp + torch.arange(0, tmp)[None, :]
            ids_ctx, idx_others = [], []
            for ii in range(N):
                ids_ctx.append((tmpm + ws_start[ii].item() * tmp + hs_start[ii].item())[:ws[ii].item(), :hs[ii].item()].flatten()[:num_seen])
                others = torch.from_numpy(np.setdiff1d(np.arange(self.patch_embed.num_patches), ids_ctx[-1].data.numpy())).long()
                others = others[torch.randperm(others.shape[0])].view(others.size())
                idx_others.append(others)

            ids_ctx = torch.stack(ids_ctx)
            idx_others = torch.stack(idx_others)
            ids_shuffle = torch.cat([ids_ctx, idx_others], 1).to(x.device)
        else:
            span = self.span[torch.randint(0, len(self.span), (1,)).item()]
            noise = torch.rand(N, L // span, device=x.device)  # noise in [0, 1]
            ids_shuffle = ((torch.argsort(noise, dim=1) * span).unsqueeze(-1) + torch.arange(0, span, device=x.device)).flatten(1).long()
            # print(span, ids_shuffle[:4, :15].data.cpu().numpy())

        # noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # print(ids_shuffle.shape)

        ids_shuffle = ids_shuffle.unsqueeze(-1).repeat(1, 1, D)
        x = torch.gather(x, dim=1, index=ids_shuffle[:, :num_seen + num_targets - 1])
        g = torch.gather(g, dim=1, index=ids_shuffle[:, num_seen:num_seen + num_targets])

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        if self.g_depth == 0:
            x_g = torch.cat([x, g], 1)
            for blk in self.blocks:
                x_g = blk(x_g, select_kv=x.shape[1], attn_mask=attn_mask)

            g = x_g[:, x.shape[1]:]
            # y_feature = x_g[:, 0, :]

            if self.one_extra_layer:
                g = self.extra_block(g, x_g[:, :x.shape[1]], attn_mask=attn_mask[num_seen + num_targets:])
        else:
            attn_mask1 = attn_mask[:num_seen + num_targets]
            attn_mask2 = attn_mask[num_seen + num_targets:]

            for lyr in range(len(self.blocks) + 1 - self.g_depth):
                x = self.blocks[lyr](x, attn_mask=attn_mask1)

            for lyr in range(len(self.blocks) + 1 - self.g_depth, len(self.blocks)):
                g = self.g_blocks[lyr + self.g_depth - len(self.blocks) - 1](g, x, attn_mask=attn_mask2)
                x = self.blocks[lyr](x, attn_mask=attn_mask1)

            g = self.g_blocks[-1](g, x, attn_mask=attn_mask2)
            # y_feature = x[:, 0, :]

        g = self.head(self.norm(g))
        # y_logits = self.head_2(self.norm_2(y_feature))
        return g, ids_shuffle #, y_logits

    def forward(self, imgs, patch_aug=False, mask_ratio=1, num_targets=None, CJ=None):
        # if isinstance(imgs, list):
        #     imgs_train = imgs[0]
        #     imgs = imgs[1]
        # else:
        #     imgs_train = imgs
        num_seen = int(self.patch_embed.num_patches * (1 - mask_ratio))
        if num_targets is None:
            num_targets = self.patch_embed.num_patches - num_seen

        attn_mask = torch.concat([torch.zeros(num_seen + 1, num_targets - 1),
                                 torch.ones(num_targets - 1, num_targets - 1).tril(),
                                 torch.ones(num_targets, num_targets - 1).tril(-1)], 0)
        attn_mask = torch.concat([
            torch.ones(num_seen + num_targets * 2, num_seen + 1), attn_mask], 1)
        attn_mask[0] = 1
        attn_mask[1:, 0] = 0
        attn_mask = attn_mask.bool().to(imgs.device)

        imgs_seq = self.patchify(imgs)

        if patch_aug:
            with torch.no_grad():
                p = self.patch_embed.patch_size[0]
                mean = torch.tensor([0.485, 0.456, 0.406]).to(imgs.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).to(imgs.device).view(1, 3, 1, 1)
                imgs_ = imgs_seq.flatten(0, 1).view(-1, p, p, 3).permute(0, 3, 1, 2).mul_(std).add_(mean)

                cj_num = 8
                patch_aug_mask = torch.empty(imgs_.shape[0] * 2, 1, 1, 1, device=imgs_.device).random_(0, int(cj_num * 1.25))
                imgs_train = torch.zeros_like(imgs_)
                imgs_train = imgs_train.addcmul_(imgs_, (patch_aug_mask[:imgs_.shape[0]] >= cj_num).type_as(imgs_))
                for i in range(cj_num):
                    imgs_train = imgs_train.addcmul_(CJ(imgs_), (patch_aug_mask[:imgs_.shape[0]] == i).type_as(imgs_))

                imgs_ = imgs_train
                patch_aug_mask = patch_aug_mask[imgs_.shape[0]:] % 4
                imgs_train = torch.zeros_like(imgs_)
                imgs_train = imgs_train.addcmul_(imgs_, (patch_aug_mask == 0).type_as(imgs_))
                for i in range(1, 4):
                    imgs_train = imgs_train.addcmul_(torch.rot90(imgs_, i, [2, 3]), (patch_aug_mask == i).type_as(imgs_))

                h = w = imgs.shape[2] // p
                imgs_train = imgs_train.sub_(mean).div_(std).view(imgs.shape[0], h, w, 3, p, p)
                imgs_train = imgs_train.permute(0, 3, 1, 4, 2, 5).flatten(2, 3).flatten(3, 4)
        else:
            imgs_train = imgs
        pred, ids_shuffle = self.forward_encoder(imgs_train, num_seen, num_targets, attn_mask)
        target_indices = ids_shuffle[:, num_seen:num_seen + num_targets]

        target = torch.gather(imgs_seq, dim=1, index=target_indices)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = ((pred - target) ** 2).mean()

        return loss, pred, target_indices#, y_logits


def xlnet_vit_base_patch16(**kwargs):
    model = XLNetViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def xlnet_vit_large_patch16(**kwargs):
    model = XLNetViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def xlnet_vit_huge_patch14(**kwargs):
    model = XLNetViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
