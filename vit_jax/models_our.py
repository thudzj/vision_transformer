# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import numpy as onp
import jax.numpy as jnp
from jax import lax
from jax import random
from vit_jax import models_resnet

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x

class DropPath(nn.Module):
  rate : float = 0.1
  @nn.compact
  def __call__(self, inputs, deterministic):
    if deterministic or self.rate == 0:
      return inputs
    else:
      keep_prob = 1 - self.rate
      rng = self.make_rng('droppath')
      broadcast_shape = list(inputs.shape)
      for dim in range(1, len(broadcast_shape)):
        broadcast_shape[dim] = 1
      mask = random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
      mask = jnp.broadcast_to(mask, inputs.shape)
      return lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(s):

    _, n_position, d_hid = s
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / onp.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = onp.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = onp.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = onp.cos(sinusoid_table[:, 1::2]) # dim 2i+1
    return jnp.expand_dims(jnp.asarray(sinusoid_table), 0)

class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  name: str = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name=self.name+"_dense1")(  # pytype: disable=wrong-arg-types
            inputs)
    x = nn.gelu(x)
    if self.dropout_rate > 0:
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name=self.name+"_dense2")(  # pytype: disable=wrong-arg-types
            x)
    if self.dropout_rate > 0:
      output = nn.Dropout(
          rate=self.dropout_rate)(
              output, deterministic=deterministic)
    return output


class Block(nn.Module):
  """Transformer layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
  """
  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  drop_path_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """
    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer block.
    """

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads)(
            x, x)
    if self.dropout_rate > 0:
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    if self.drop_path_rate > 0:
      x = DropPath(rate=self.drop_path_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)
    if self.drop_path_rate > 0:
      y = DropPath(rate=self.drop_path_rate)(y, deterministic=deterministic)
    return x + y


class Block2S(nn.Module):
  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  drop_path_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, g_inputs, m1, m2, *, deterministic):
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    assert g_inputs.ndim == 3, f'Expected (batch, seq, hidden) got {g_inputs.shape}'

    g = nn.LayerNorm(dtype=self.dtype, name='two_stream_ln1')(g_inputs)
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    
    g = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
        name='two_stream_attn')(
            g, x, mask=m2)

    if self.dropout_rate > 0:
      g = nn.Dropout(rate=self.dropout_rate)(g, deterministic=deterministic)
    if self.drop_path_rate > 0:
      g = DropPath(rate=self.drop_path_rate)(g, deterministic=deterministic)
    g = g + g_inputs

    h = nn.LayerNorm(dtype=self.dtype, name='two_stream_ln2')(g)
    h = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, 
        dropout_rate=self.dropout_rate, name='two_stream_mlp')(
            h, deterministic=deterministic)
    if self.drop_path_rate > 0:
      h = DropPath(rate=self.drop_path_rate)(h, deterministic=deterministic)


    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads)(
            x, x, mask=m1)
    if self.dropout_rate > 0:
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    if self.drop_path_rate > 0:
      x = DropPath(rate=self.drop_path_rate)(x, deterministic=deterministic)
    x = x + inputs

    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)
    if self.drop_path_rate > 0:
      y = DropPath(rate=self.drop_path_rate)(y, deterministic=deterministic)
    return x + y, g + h

class LastBlock2S(nn.Module):
  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  drop_path_rate: float = 0.1

  @nn.compact
  def __call__(self, x, g_inputs, m2, *, deterministic):
    assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'
    assert g_inputs.ndim == 3, f'Expected (batch, seq, hidden) got {g_inputs.shape}'

    g = nn.LayerNorm(dtype=self.dtype, name='two_stream_ln1')(g_inputs)
    g = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
        name='two_stream_attn')(
            g, x,  mask=m2)
    if self.dropout_rate > 0:
      g = nn.Dropout(rate=self.dropout_rate)(g, deterministic=deterministic)
    if self.drop_path_rate > 0:
      g = DropPath(rate=self.drop_path_rate)(g, deterministic=deterministic)
    g = g + g_inputs

    h = nn.LayerNorm(dtype=self.dtype, name='two_stream_ln2')(g)
    h = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, 
        dropout_rate=self.dropout_rate, name='two_stream_mlp')(
            h, deterministic=deterministic)
    if self.drop_path_rate > 0:
      h = DropPath(rate=self.drop_path_rate)(h, deterministic=deterministic)
    return g + h


class Encoder(nn.Module):

  patches: Any
  hidden_size: int
  num_layers: int
  mlp_dim: int
  num_heads: int
  num_mask: int = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  drop_path_rate: float = 0.1
  classifier: str = None
  two_stream: bool = False

  @nn.compact
  def __call__(self, inputs, *, train, masks=None):
    # We can merge s2d+emb into a single conv; it's the same.
    x = nn.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding')(
            inputs)

    # Here, x is a grid of embeddings.
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    pe = get_sinusoid_encoding_table((1, x.shape[1], c))
    x = x + pe
    if self.dropout_rate > 0:
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    dpr = onp.linspace(0, self.drop_path_rate, self.num_layers)
    if not self.two_stream:
      if self.num_mask:
        assert masks is not None
        x = jnp.take_along_axis(x, jnp.expand_dims(masks[:, :-self.num_mask], -1), 1)

      for lyr in range(self.num_layers):
        x = Block(
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            drop_path_rate=dpr[lyr],
            name=f'encoderblock_{lyr}',
            num_heads=self.num_heads)(
                x, deterministic=not train)
      encoded = nn.LayerNorm(name='encoder_norm')(x)
      return encoded
    else:
      assert masks is not None
      x = jnp.take_along_axis(x, jnp.expand_dims(masks, -1), 1)

      g = self.param('two_stream_g', nn.initializers.normal(stddev=0.02), (1, 1, c))
      g = jnp.take_along_axis(g + pe, jnp.expand_dims(masks[:, -self.num_mask:], -1), 1)

      m1 = jnp.concatenate([jnp.zeros((x.shape[1] - self.num_mask, self.num_mask)), 
                            jnp.tril(jnp.ones((self.num_mask, self.num_mask)))], axis=0)
      m1 = jnp.concatenate([jnp.ones((x.shape[1], x.shape[1] - self.num_mask)), m1], axis=1)
      m1 = jnp.expand_dims(jnp.expand_dims(m1.astype(bool), 0), 0)
      # m1 = m1.astype(bool)
      # m1 = jnp.tile(m1, (n, self.num_heads, 1, 1))

      m2 = jnp.tril(jnp.ones((self.num_mask, self.num_mask)), k=-1)
      m2 = jnp.concatenate([jnp.ones((self.num_mask, x.shape[1] - self.num_mask)), m2], axis=1)
      m2 = jnp.expand_dims(jnp.expand_dims(m2.astype(bool), 0), 0)
      # m2 = m2.astype(bool)
      # m2 = jnp.tile(m2, (n, self.num_heads, 1, 1))
      for lyr in range(self.num_layers):
        x, g = Block2S(
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            drop_path_rate=dpr[lyr],
            name=f'encoderblock_{lyr}',
            num_heads=self.num_heads)(
                x, g, m1, m2, deterministic=not train)

      encoded = nn.LayerNorm(name='encoder_norm')(x)

      g = LastBlock2S(
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            drop_path_rate=0,
            name=f'encoderblock_{self.num_layers}',
            num_heads=self.num_heads)(
                encoded, g, m2, deterministic=not train)
      g_encoded = nn.LayerNorm(name='two_stream_lastln')(g)
      return g_encoded
    

class Decoder(nn.Module):

  hidden_size: int
  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  drop_path_rate: float = 0.1
  out_dim: int = 768

  @nn.compact
  def __call__(self, inputs, *, train, return_token_num):
    assert inputs.ndim == 3  # (batch, len, emb)

    x = inputs
    dpr = onp.linspace(0, self.drop_path_rate, self.num_layers)
    for lyr in range(self.num_layers):
      x = Block(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          drop_path_rate=dpr[lyr],
          name=f'decoderblock_{lyr}',
          num_heads=self.num_heads)(
              x, deterministic=not train)

    if return_token_num > 0:
        x = x[:, -return_token_num:]

    x = nn.LayerNorm(name='decoder_norm')(x)
    decoded = nn.Dense(
        name='decoder_out',
        features=self.out_dim,
        dtype=jnp.float32,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(x)
    return decoded


class MAE(nn.Module):
  """Masked Auto-Encoder."""

  num_mask: int
  encoder: Any
  decoder: Any

  @nn.compact
  def __call__(self, inputs, masks, *, train):

    x_vis = Encoder(name='Encoder', num_mask=self.num_mask, **self.encoder)(
                                               inputs, masks=masks, train=train)
    x_vis = nn.Dense(
        name='Encoder2Decoder',
        features=self.decoder.hidden_size,
        dtype=jnp.float32,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(x_vis)

    B, N, C = x_vis.shape
    expand_pos_embed = jnp.tile(get_sinusoid_encoding_table((1, N, C)), [B, 1, 1])

    # we don't unshuffle the correct visible token order,
    # but shuffle the pos embedding accorddingly.
    pos_emd_vis = jnp.take_along_axis(expand_pos_embed, jnp.expand_dims(masks[:, :-self.num_mask], -1), 1)
    pos_emd_mask = jnp.take_along_axis(expand_pos_embed, jnp.expand_dims(masks[:, -self.num_mask:], -1), 1)

    mt = self.param('mask_token', nn.initializers.normal(stddev=1e-6), (1, 1, C))
    x_full = jnp.concatenate([x_vis + pos_emd_vis, mt + pos_emd_mask], 1)

    # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
    x = Decoder(name='Decoder', **self.decoder)(x_full, train=train, return_token_num=self.num_mask)
    return x


class XLNet(nn.Module):
  """XLNet"""

  num_mask: int
  encoder: Any
  out_dim: int

  @nn.compact
  def __call__(self, inputs, masks, *, train):
    g = Encoder(name='Encoder', num_mask=self.num_mask, **self.encoder)(
                                              inputs, masks=masks, train=train)
    g = nn.Dense(
        name='out',
        features=self.out_dim,
        dtype=jnp.float32,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(g)
    return g


class ViT(nn.Module):
  """ViT."""
  encoder: Any
  classifier: str = 'gap'
  num_classes: int = 1000
  representation_size: int = None

  @nn.compact
  def __call__(self, inputs, *, train):

    x = Encoder(name='Encoder', classifier=self.classifier, **self.encoder)(inputs, train=train)

    if self.classifier == 'token':
      x = x[:, 0]
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')

    if self.representation_size is not None:
      x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = IdentityLayer(name='pre_logits')(x)

    if self.num_classes:
      x = nn.Dense(
        features=self.num_classes,
        name='head',
        kernel_init=nn.initializers.zeros)(x)
    return x
