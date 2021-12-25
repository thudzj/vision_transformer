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

from typing import Any, Dict, Iterable, Tuple, Union

import ml_collections

def get_b16_config():
  """Returns the ViT-B/16 configuration."""
  config = ml_collections.ConfigDict()
  config.name = 'ViT-B_16'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 768
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 3072
  config.transformer.num_heads = 12
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = 'token'
  config.representation_size = None
  return config

def get_l16_config():
  """Returns the ViT-L/16 configuration."""
  config = ml_collections.ConfigDict()
  config.name = 'ViT-L_16'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 1024
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 4096
  config.transformer.num_heads = 16
  config.transformer.num_layers = 24
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None
  return config

def get_h14_config():
  """Returns the ViT-H/14 configuration."""
  config = ml_collections.ConfigDict()
  config.name = 'ViT-H_14'
  config.patches = ml_collections.ConfigDict({'size': (14, 14)})
  config.hidden_size = 1280
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 5120
  config.transformer.num_heads = 16
  config.transformer.num_layers = 32
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None
  return config

def get_config(model):

  config = ml_collections.ConfigDict()

  # Where to search for pretrained ViT models.
  # Can be downloaded from gs://vit_models/imagenet21k
  # config.pretrained_dir = None
  # Which dataset to finetune on. This can be the name of a tfds dataset
  # (see https://www.tensorflow.org/datasets/catalog/overview), or the path to
  # a directory with the following structure ($filename can be arbitrary):
  # "{train,test}/$class_name/$filename.jpg"
  config.dataset = '/data/LargeData/Large/ImageNet/'
  # Path to manually downloaded dataset
  config.tfds_manual_dir = None
  # Path to tensorflow_datasets directory
  config.tfds_data_dir = None
  # Number of steps; determined by hyper module if not specified.
  config.total_steps = 50_000

  # Resizes global gradients.
  config.grad_norm_clip = 1.0
  # Datatype to use for momentum state ("bfloat16" or "float32").
  config.optim_dtype = 'bfloat16'
  # Accumulate gradients over multiple steps to save on memory.
  config.accum_steps = 8

  # Batch size for training.
  config.batch = 512
  # Batch size for evaluation.
  config.batch_eval = 512
  # Shuffle buffer size.
  config.shuffle_buffer = 50_000
  # Run prediction on validation set every so many steps
  config.eval_every = 100
  # Log progress every so many steps.
  config.progress_every = 10
  # How often to write checkpoints. Specifying 0 disables checkpointing.
  config.checkpoint_every = 1_000

  # Number of batches to prefetch to device.
  config.prefetch = 2

  # Base learning-rate for fine-tuning.
  config.base_lr = 0.03
  # How to decay the learning rate ("cosine" or "linear").
  config.decay_type = 'cosine'
  # How to decay the learning rate.
  config.warmup_steps = 500

  # Alternatives : inference_time.
  config.trainer = 'train'

  config.pp = ml_collections.ConfigDict(
                 {'train': 'train',
                  'test': 'val',
                  'crop': 224})

  get_model_config = eval(f'get_{model}_config')
  config.model = get_model_config()

  return config.lock()


