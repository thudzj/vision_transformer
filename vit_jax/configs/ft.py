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
  config.half_precision = True

  config.encoder = ml_collections.ConfigDict()
  config.encoder.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.encoder.hidden_size = 768
  config.encoder.mlp_dim = 3072
  config.encoder.num_heads = 12
  config.encoder.num_layers = 12
  config.encoder.attention_dropout_rate = 0.0
  config.encoder.dropout_rate = 0.0
  config.encoder.drop_path_rate = 0.1

  config.classifier = 'gap'
  config.representation_size = None
  return config

def get_l16_config():
  """Returns the ViT-L/16 configuration."""
  config = ml_collections.ConfigDict()
  config.name = 'ViT-L_16'
  config.half_precision = True

  config.encoder = ml_collections.ConfigDict()
  config.encoder.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.encoder.hidden_size = 1024
  config.encoder.mlp_dim = 4096
  config.encoder.num_heads = 16
  config.encoder.num_layers = 24
  config.encoder.attention_dropout_rate = 0.0
  config.encoder.dropout_rate = 0.0
  config.encoder.drop_path_rate = 0.1

  config.classifier = 'gap'
  config.representation_size = None
  return config

# def get_h14_config():
#   """Returns the ViT-H/14 configuration."""
#   config = ml_collections.ConfigDict()
#   config.name = 'ViT-H_14'
#   config.patches = ml_collections.ConfigDict({'size': (14, 14)})
#   config.hidden_size = 1280
#   config.transformer = ml_collections.ConfigDict()
#   config.transformer.mlp_dim = 5120
#   config.transformer.num_heads = 16
#   config.transformer.num_layers = 32
#   config.transformer.attention_dropout_rate = 0.0
#   config.transformer.dropout_rate = 0.1
#   config.transformer.use_learnable_pos_emb=False
#   config.classifier = 'token'
#   config.representation_size = None
#   return config

def get_config(model):

  config = ml_collections.ConfigDict()
  config.trainer = 'finetune'
  
  config.pretrained_path = ''
  config.dataset = '/data/LargeData/Large/ImageNet/'

  config.weight_decay = 0.05
  config.base_lr = 1e-3
  config.decay_type = 'cosine'
  config.warmup_epochs = 5
  config.epochs = 100

  config.layer_wise_lr_decay = 0.75

  config.beta1 = 0.9
  config.beta2 = 0.999

  config.label_smoothing = 0.1
  config.mix_prob = 1.
  config.switch_prob = 0.5
  config.mixup = 0.8
  config.cutmix = 1.
  config.flip = True
  config.randaug = '2-9-0.5'

  # Batch size for training.
  config.batch = 512
  # Batch size for evaluation.
  config.batch_eval = 512
  # Shuffle buffer size.
  config.shuffle_buffer = 50_000
  # Run prediction on validation set every so many steps
  config.eval_every = 1000
  # Log progress every so many steps.
  config.progress_every = 100
  # How often to write checkpoints. Specifying 0 disables checkpointing.
  config.checkpoint_every = 10_000

  # Number of batches to prefetch to device.
  config.prefetch = 2

  config.optim_half_precision = False
  config.dynamic_scale = True

  # Base learning-rate for fine-tuning.
  # config.base_lr = 0.03
  # How to decay the learning rate ("cosine" or "linear").

  # How to decay the learning rate.
  # config.warmup_steps = 500

  config.pp = ml_collections.ConfigDict(
                 {'train': 'train',
                  'test': 'val',
                  'crop': 224})

  get_model_config = eval(f'get_{model}_config')
  config.model = get_model_config()

  config.patch_size = config.model.encoder.patches['size'][0]
  config.num_patches = (config.pp['crop'] // config.model.encoder.patches['size'][0])**2

  return config.lock()
