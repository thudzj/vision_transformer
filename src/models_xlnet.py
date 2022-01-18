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

from timm.models.vision_transformer import PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

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
                 norm_pix_loss=False, pred_pos=False, pred_pos_smoothing=0.,
                 g_depth=0):
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])


        if g_depth > 0:
            self.g_blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                for i in range(g_depth)])

        self.norm = norm_layer(embed_dim)
        if pred_pos:
            self.head = nn.Linear(embed_dim, num_patches, bias=True)
        else:
            self.head = nn.Linear(embed_dim, patch_size**2 * in_chans, bias=True)

        if not pred_pos:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            torch.nn.init.normal_(self.mask_token, std=.02)
        else:
            self.mask_token = None
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.pred_pos = pred_pos
        self.pred_pos_smoothing = pred_pos_smoothing
        self.g_depth = g_depth

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

        if self.mask_token is None:
            g = x_
        else:
            g = (self.pos_embed[:, 1:, :] + self.mask_token).expand_as(x_)

        # permutation auto-regressive modeling
        N, L, D = x.shape  # batch, length, dim
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
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
        else:
            attn_mask1 = attn_mask[:num_seen + num_targets]
            attn_mask2 = attn_mask[num_seen + num_targets:]

            for lyr in range(len(self.blocks) + 1 - self.g_depth):
                x = self.blocks[lyr](x, attn_mask=attn_mask1)

            for lyr in range(len(self.blocks) + 1 - self.g_depth, len(self.blocks)):
                g = self.g_blocks[lyr + self.g_depth - len(self.blocks) - 1](g, x, attn_mask=attn_mask2)
                x = self.blocks[lyr](x, attn_mask=attn_mask1)

            g = self.g_blocks[-1](g, x, attn_mask=attn_mask2)

        g = self.norm(g)
        g = self.head(g)
        return g, ids_shuffle

    def forward(self, imgs, mask_ratio=1, num_targets=None, attn_mask=None):
        num_seen = int(self.patch_embed.num_patches * (1 - mask_ratio))
        if num_targets is None:
            num_targets = self.patch_embed.num_patches - num_seen

        pred, ids_shuffle = self.forward_encoder(imgs, num_seen, num_targets, attn_mask)
        target_indices = ids_shuffle[:, num_seen:num_seen + num_targets]

        if self.pred_pos:
            W = int(self.patch_embed.num_patches**.5)
            # perform label smoothing
            row_labels = torch.div(target_indices[:,:,0].reshape(-1, 1, 1), W, rounding_mode='trunc')
            col_labels = target_indices[:,:,0].reshape(-1, 1, 1) % W
            new_labels = torch.exp(-((row_labels - torch.arange(W, device=imgs.device).view(1, -1, 1)) ** 2 +
                            (col_labels - torch.arange(W, device=imgs.device).view(1, 1, -1)) ** 2)
                          / 2. / (self.pred_pos_smoothing + 1e-8))
            target = new_labels.view(target_indices.shape[0], target_indices.shape[1], -1)
            target = target / torch.sum(target, dim=-1, keepdim=True)

            # ce loss
            loss = - torch.sum(pred.log_softmax(-1) * target, dim=-1).mean()
        else:
            target = torch.gather(self.patchify(imgs), dim=1, index=target_indices)
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5

            loss = ((pred - target) ** 2).mean()
        return loss, pred, target_indices


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
