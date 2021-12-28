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
from vit_jax import models_our
# from vit_jax import momentum_clip
from vit_jax import utils

from vit_jax.input_pipeline import MEAN_RGB, STDDEV_RGB

def _filter(patterns, x, _):
  for p in patterns:
    if p in x:
      return True
  return False

def make_update_fn(*, apply_fn, lr_fn, label_smoothing, mix_prob, switch_prob, 
                   mixup, cutmix, lr_multipliers):
  """Returns update step for data parallel training."""

  def update_fn(opt, step, batch, rng):

    rng1, rng2, new_rng = jax.random.split(rng, 3)
    # Bind the rng key to the device id (which is unique across hosts)
    # Note: This is only used for multi-host training (i.e. multiple computers
    # each with multiple accelerators).
    dropout_rng = jax.random.fold_in(rng1, jax.lax.axis_index('batch'))
    droppath_rng = jax.random.fold_in(rng2, jax.lax.axis_index('batch'))

    def mixup_cutmix_labelsmootning(images, labels):
      if mixup > 0. and cutmix > 0.:
          use_cutmix = (jax.random.uniform(rng) < switch_prob).astype(float)
          lam_mix = jax.random.beta(rng, cutmix, cutmix) * use_cutmix + \
              jax.random.beta(rng, mixup, mixup) * (1 - use_cutmix)
      elif mixup > 0.:
          use_cutmix = 0
          lam_mix = jax.random.beta(rng, mixup, mixup)
      elif cutmix > 0.:
          use_cutmix = 1
          lam_mix = jnp.random.beta(rng, cutmix, cutmix)
      mask1 = (jax.random.uniform(rng) < mix_prob).astype(float)
      lam = mask1 * lam_mix + (1 - mask1)
      use_cutmix = mask1 * use_cutmix

      flipped_images = jnp.flip(images, axis=0)
      mixup_images = images * lam + flipped_images * (1. - lam)

      (yl, yh, xl, xh), cutmix_lam = utils.cutmix_bbox_and_lam(images.shape, lam, rng)
      ym = jnp.arange(images.shape[1])
      ym = (ym >= yl).astype(float) * (ym < yh).astype(float)
      xm = jnp.arange(images.shape[2])
      xm = (xm >= xl).astype(float) * (xm < xh).astype(float)
      cutmix_mask = jnp.tile((jnp.expand_dims(ym, -1) * jnp.expand_dims(xm, 0)).reshape(
        1, images.shape[1], images.shape[2], 1), (images.shape[0], 1, 1, images.shape[-1]))
      cutmix_images = images * (1 - cutmix_mask) + flipped_images * cutmix_mask

      images = use_cutmix * cutmix_images + (1 - use_cutmix) * mixup_images
      lam = use_cutmix * cutmix_lam + (1 - use_cutmix) * lam

      labels = utils.mixup_target(labels, lam, label_smoothing)
      return images, labels

    def cross_entropy_loss(*, logits, labels):
      logp = jax.nn.log_softmax(logits)
      return -jnp.mean(jnp.sum(logp * labels, axis=1))

    def loss_fn(params, images, labels):
      logits = apply_fn(
          dict(params=params),
          rngs=dict(dropout=dropout_rng, droppath=droppath_rng),
          inputs=images,
          train=True)
      return cross_entropy_loss(logits=logits, labels=labels)
    
    images, labels = mixup_cutmix_labelsmootning(batch['image'], batch['label'])
    l, g = jax.value_and_grad(loss_fn)(opt.target, images, labels)
    g = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), g)
    l = jax.lax.pmean(l, axis_name='batch')
    hparams = [item.replace(learning_rate=lr_fn(step) * lr_multipliers[i]) 
                      for i, item in enumerate(opt.optimizer_def.hyper_params)]
    opt = opt.apply_gradient(g, hyper_params=hparams)
    return opt, l, new_rng, images
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

  # Build ViT
  model = models_our.ViT(**config.model)
  def init_model():
    return model.init(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension for initialization.
        jnp.ones(batch['image'].shape[1:], batch['image'].dtype.name),
        train=False)

  # Use JIT to make sure params reside in CPU memory.
  variables = jax.jit(init_model, backend='cpu')()

  if not config.pretrained_path or not tf.io.gfile.exists(config.pretrained_path):
    logging.info(f'Could not find "{config.pretrained_path}"')
    params = variables['params']
  else:
    logging.info(f'Load pretrained encoder from "{config.pretrained_path}"')
    state = flax_checkpoints.restore_checkpoint(config.pretrained_path, None)
    if 'mae' in config.trainer:
      params = {'Encoder': state['0']['target']['Encoder'], 'head': variables['params']['head']}
    elif 'xlnet' in config.trainer:
      # todo
      assert False
    else:
      assert False
    
    if config.model.representation_size is not None:
      params['pre_logits'] = variables['params']['pre_logits']
    params = flax.core.freeze(params)

  total_steps = config.epochs * steps_per_epoch
  lr_fn = utils.create_learning_rate_schedule(total_steps, 
                                              config.base_lr * config.batch / 256.,
                                              config.decay_type,
                                              config.warmup_epochs * steps_per_epoch)
  
  lr_multipliers = list(config.layer_wise_lr_decay ** 
    (config.model.encoder.num_layers + 1 - i) for i in range(
      config.model.encoder.num_layers + 2))
  # print(lr_multipliers)

  update_fn_repl = make_update_fn(
      apply_fn=model.apply, lr_fn=lr_fn, 
      label_smoothing=config.label_smoothing, 
      mix_prob=config.mix_prob,
      switch_prob=config.switch_prob,
      mixup=config.mixup,
      cutmix=config.cutmix,
      lr_multipliers=lr_multipliers)
  infer_fn_repl = jax.pmap(functools.partial(model.apply, train=False))

  # # Create optimizer and replicate it over all TPUs/GPUs
  # opt = flax.optim.Adam(beta1=config.beta1, beta2=config.beta2, 
  #                       weight_decay=config.weight_decay).create(params)
  groups = [(flax.optim.ModelParamTraversal(
               functools.partial(_filter, ['/embedding/'])), 
             flax.optim.Adam(beta1=config.beta1, beta2=config.beta2, 
                             weight_decay=config.weight_decay))]
  for i in range(config.model.encoder.num_layers):
    groups.append((flax.optim.ModelParamTraversal(
                     functools.partial(_filter, ['/encoderblock_{}/'.format(i)])), 
                   flax.optim.Adam(beta1=config.beta1, beta2=config.beta2, 
                                   weight_decay=config.weight_decay)))
  groups.append((flax.optim.ModelParamTraversal(
                   functools.partial(_filter, ['/encoder_norm/', '/pre_logits/', '/head/'])),
                 flax.optim.Adam(beta1=config.beta1, beta2=config.beta2, 
                                 weight_decay=config.weight_decay)))
  opt_def = flax.optim.MultiOptimizer(*groups)
  opt = opt_def.create(params)

  cnt = 0
  for ii, item in enumerate(opt_def.traversals):
    for k in item.iterate(params):
      cnt += 1
  cnt_ = 0
  dummy = flax.optim.ModelParamTraversal(lambda path, _: True)
  for k in dummy.iterate(params):
    cnt_ += 1  
  assert(cnt_ == cnt)
  
  initial_step = 1
  # opt, initial_step = flax_checkpoints.restore_checkpoint(
  #     workdir, (opt, initial_step))
  # logging.info('Will start/continue training at initial_step=%d', initial_step)

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
      opt_repl, loss_repl, update_rng_repl, images_ = update_fn_repl(
          opt_repl, flax.jax_utils.replicate(step), batch, update_rng_repl)

    # check if cutmix works
    if step < 50:
        images_ = np.asarray(images_, 
            images_.dtype.name)[:, :3].reshape(-1, 224, 224, 3)
        unnorm_images = images_ * np.array(STDDEV_RGB)/255. + np.array(MEAN_RGB)/255. 
        writer.write_images(
          step,
          dict(
              training_samples=unnorm_images)
          )

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

      accuracies = []
      lt0 = time.time()
      for test_batch in input_pipeline.prefetch(ds_test, config.prefetch):
        logits = infer_fn_repl(
            dict(params=opt_repl.target), test_batch['image'])
        accuracies.append(
            (np.argmax(logits,
                       axis=-1) == np.argmax(test_batch['label'],
                                             axis=-1)).mean())
      accuracy_test = np.mean(accuracies)
      img_sec_core_test = (
          config.batch_eval * ds_test.cardinality().numpy() /
          (time.time() - lt0) / jax.device_count())
      lt0 = time.time()

      lr = float(lr_fn(step))
      logging.info(f'Step: {step} '  # pylint: disable=logging-format-interpolation
                   f'Learning rate: {lr:.7f}, '
                   f'Test accuracy: {accuracy_test:0.5f}, '
                   f'img/sec/core: {img_sec_core_test:.1f}')
      writer.write_scalars(
          step,
          dict(
              accuracy_test=accuracy_test,
              lr=lr,
              img_sec_core_test=img_sec_core_test))

    # Store checkpoint.
    if ((config.checkpoint_every and step % config.checkpoint_every == 0) or
        step == total_steps):
      checkpoint_path = flax_checkpoints.save_checkpoint(
          workdir, (flax.jax_utils.unreplicate(opt_repl), step), step)
      logging.info('Stored checkpoint at step %d to "%s"', step,
                   checkpoint_path)

  return flax.jax_utils.unreplicate(opt_repl)
