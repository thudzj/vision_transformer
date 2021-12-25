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

import functools
import os
import time

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax.training import checkpoints as flax_checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf

# from vit_jax import checkpoint
from vit_jax import input_pipeline
from vit_jax import mae
# from vit_jax import momentum_clip
from vit_jax import utils

from vit_jax.input_pipeline import MEAN_RGB, STDDEV_RGB
from einops import rearrange

def preprocess(batch, normlize_target, patch_size, num_mask):
  images = batch['image']
  masks = batch['label']

  mean = jnp.array(MEAN_RGB)/255.
  std = jnp.array(STDDEV_RGB)/255.
  unnorm_images = images * std + mean  # in [0, 1]
  if normlize_target:
      images_squeeze = rearrange(unnorm_images, 'b (h p1) (w p2) c -> b (h w) (p1 p2) c', 
                                 p1=patch_size, p2=patch_size)
      images_norm = (images_squeeze - jnp.mean(images_squeeze, axis=-2, keepdims=True)
          ) / (jnp.sqrt(jnp.nanvar(images_squeeze, axis=-2, ddof=1, keepdims=True)) + 1e-6)
      # we find that the mean is about 0.48 and standard deviation is about 0.08.
      images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
  else:
      images_patch = rearrange(unnorm_images, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', 
                               p1=patch_size, p2=patch_size)
  labels = jnp.take_along_axis(images_patch, jnp.expand_dims(masks[:, -num_mask:], -1), 1)
  return images, masks, labels

def make_update_fn(*, apply_fn, normlize_target, patch_size, num_mask, lr_fn):
  """Returns update step for data parallel training."""

  def update_fn(opt, step, batch, rng):

    _, rng1 = jax.random.split(rng)
    _, new_rng = jax.random.split(rng1)
    # Bind the rng key to the device id (which is unique across hosts)
    # Note: This is only used for multi-host training (i.e. multiple computers
    # each with multiple accelerators).
    dropout_rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
    droppath_rng = jax.random.fold_in(rng1, jax.lax.axis_index('batch'))

    def mse_loss(*, logits, labels):
      return jnp.mean((logits - labels) ** 2)

    def loss_fn(params, images, masks, labels):
      logits = apply_fn(
          dict(params=params),
          rngs=dict(dropout=dropout_rng, droppath=droppath_rng),
          inputs=images,
          masks=masks,
          train=True)
      return mse_loss(logits=logits, labels=labels)
    
    images, masks, labels = preprocess(batch, normlize_target, patch_size, num_mask)
    l, g = jax.value_and_grad(loss_fn)(opt.target, images, masks, labels)
    g = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), g)
    l = jax.lax.pmean(l, axis_name='batch')
    opt = opt.apply_gradient(g, learning_rate=lr_fn(step))
    return opt, l, new_rng

  return jax.pmap(update_fn, axis_name='batch', donate_argnums=(0,))


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Runs training interleaved with evaluation."""

  # Setup input pipeline
  dataset_info = input_pipeline.get_dataset_info(config.dataset, 'train')
  steps_per_epoch = (
      dataset_info['num_examples'] // config.batch
  )

  ds_train, ds_test = input_pipeline.get_datasets(config)
  batch = next(iter(ds_train))
  logging.info(ds_train)
  logging.info(ds_test)

  # for step, batch in zip(
  #     range(0, 10 + 1),
  #     input_pipeline.prefetch(ds_train, config.prefetch)):
  #     print(batch['image'].shape, batch['image'].dtype, np.max(batch['image']), np.min(batch['image']))
  #     mean = jnp.array(MEAN_RGB)/255.
  #     std = jnp.array(STDDEV_RGB)/255.
  #     unnorm_images = batch['image'] * std + mean  # in [0, 1]
  #     print(np.max(unnorm_images), np.min(unnorm_images))
  #     if step == 5:
  #       exit()

  # Build MAE
  model = mae.MAE(num_mask=config.num_mask, **config.model)

  def init_model():
    return model.init(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension for initialization.
        jnp.ones(batch['image'].shape[1:], batch['image'].dtype.name),
        np.asarray(memoryview(batch['label']), batch['label'].dtype.name)[0],
        train=False)

  # Use JIT to make sure params reside in CPU memory.
  variables = jax.jit(init_model, backend='cpu')()
  params = variables['params']

  total_steps = config.epochs * steps_per_epoch
  lr_fn = utils.create_learning_rate_schedule(total_steps, 
                                              config.base_lr * config.batch / 256.,
                                              config.decay_type,
                                              config.warmup_epochs * steps_per_epoch)

  update_fn_repl = make_update_fn(
      apply_fn=model.apply, normlize_target=config.normlize_target, 
      patch_size=config.patch_size, 
      num_mask=config.num_mask, lr_fn=lr_fn)
  infer_fn_repl = jax.pmap(functools.partial(model.apply, train=False))

  # Create optimizer and replicate it over all TPUs/GPUs
  opt = flax.optim.Adam(beta1=config.beta1, beta2=config.beta2, 
                        weight_decay=config.weight_decay).create(params)
  initial_step = 1
  opt_repl = flax.jax_utils.replicate(opt)

  # Delete references to the objects that are not needed anymore
  del opt
  del params

  # Prepare the learning-rate and pre-fetch it to device to avoid delays.
  update_rng_repl = flax.jax_utils.replicate(jax.random.PRNGKey(0))

  # Setup metric writer & hooks.
  writer = metric_writers.create_default_writer(workdir, asynchronous=False)
  writer.write_hparams(config.to_dict())
  hooks = [
      periodic_actions.Profile(logdir=workdir),
      periodic_actions.ReportProgress(
          num_train_steps=total_steps, writer=writer),
  ]

  # Run training loop
  logging.info('Starting training loop; initial compile can take a while...')
  t0 = lt0 = time.time()
  lstep = initial_step
  for step, batch in zip(
      range(initial_step, total_steps + 1),
      input_pipeline.prefetch(ds_train, config.prefetch)):

    with jax.profiler.StepTraceContext('train', step_num=step):
      opt_repl, loss_repl, update_rng_repl = update_fn_repl(
          opt_repl, flax.jax_utils.replicate(step), batch, update_rng_repl)

    for hook in hooks:
      hook(step)

    if step == initial_step:
      logging.info('First step took %.1f seconds.', time.time() - t0)
      t0 = time.time()
      lt0, lstep = time.time(), step

    # Report training metrics
    if config.progress_every and step % config.progress_every == 0:
      img_sec_core_train = (config.batch * (step - lstep) /
                            (time.time() - lt0)) / jax.device_count()
      lt0, lstep = time.time(), step
      writer.write_scalars(
          step,
          dict(
              train_loss=float(flax.jax_utils.unreplicate(loss_repl)),
              img_sec_core_train=img_sec_core_train))
      done = step / total_steps
      logging.info(f'Step: {step}/{total_steps} {100*done:.1f}%, '  # pylint: disable=logging-format-interpolation
                   f'img/sec/core: {img_sec_core_train:.1f}, '
                   f'ETA: {(time.time()-t0)/done*(1-done)/3600:.2f}h')

    # Run evaluation
    if ((config.eval_every and step % config.eval_every == 0) or
        (step == total_steps)):
      test_batch = next(iter(ds_test))
      images = np.asarray(memoryview(test_batch['image']), test_batch['image'].dtype.name)
      masks = np.asarray(memoryview(test_batch['label']), test_batch['label'].dtype.name)

      mean = np.array(MEAN_RGB)/255.
      std = np.array(STDDEV_RGB)/255.
      unnorm_images = images * std + mean  # in [0, 1]
      unnorm_images = rearrange(unnorm_images, 'a b (h p1) (w p2) c -> a b (h w) (p1 p2) c', 
                                p1=config.patch_size, p2=config.patch_size)
      if config.normlize_target:
          patches_mean = jnp.mean(unnorm_images, axis=-2, keepdims=True)
          patches_std = jnp.sqrt(jnp.nanvar(unnorm_images, axis=-2, ddof=1, keepdims=True))
      else:
          patches_mean = 0
          patches_std = 1
      recon = infer_fn_repl(dict(params=opt_repl.target), inputs=images, masks=masks)
      recon = rearrange(recon[0], 'b n (p c) -> b n p c', c=3)
      recon = recon * jnp.take_along_axis(patches_std[0],
                  jnp.expand_dims(jnp.expand_dims(masks[0, :, -config.num_mask:], -1), -1), 1) \
              + jnp.take_along_axis(patches_mean[0], 
                  jnp.expand_dims(jnp.expand_dims(masks[0, :, -config.num_mask:], -1), -1), 1)

      vis = jnp.take_along_axis(unnorm_images[0], jnp.expand_dims(
          jnp.expand_dims(masks[0, :, :-config.num_mask], -1), -1), 1)
      recon, vis = jnp.concatenate([vis, recon], axis=1), jnp.concatenate([vis, jnp.zeros_like(recon)], axis=1)

      unnorm_images = rearrange(unnorm_images[0], 'b (h w) (p1 p2) c -> b (h p1) (w p2) c', 
                                p1=config.patch_size, h=config.pp['crop']//config.patch_size)
      
      reverse_masks = jnp.expand_dims(jnp.expand_dims(jnp.argsort(masks[0]), -1), -1)
      recon = jnp.take_along_axis(recon, reverse_masks, 1)
      vis = jnp.take_along_axis(vis, reverse_masks, 1)
      recon = rearrange(recon, 'b (h w) (p1 p2) c -> b (h p1) (w p2) c', 
                        p1=config.patch_size, h=config.pp['crop']//config.patch_size)
      vis = rearrange(vis, 'b (h w) (p1 p2) c -> b (h p1) (w p2) c', 
                      p1=config.patch_size, h=config.pp['crop']//config.patch_size)
      writer.write_images(
          step,
          dict(
              samples=unnorm_images,
              vis=vis,
              recon=recon,)
          )

    # Store checkpoint.
    if ((config.checkpoint_every and step % config.checkpoint_every == 0) or
        step == total_steps):
      checkpoint_path = flax_checkpoints.save_checkpoint(
          workdir, (flax.jax_utils.unreplicate(opt_repl), step), step)
      logging.info('Stored checkpoint at step %d to "%s"', step,
                   checkpoint_path)

  return flax.jax_utils.unreplicate(opt_repl)
