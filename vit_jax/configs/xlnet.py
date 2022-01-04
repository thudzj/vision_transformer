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
  config.encoder.drop_path_rate = 0.0
  config.encoder.two_stream = 10

  config.encoder.g_mlp_dim = 3072
  config.encoder.g_num_heads = 12
  config.encoder.g_attention_dropout_rate = 0.0
  config.encoder.g_dropout_rate = 0.0
  config.encoder.g_predict_pos = False
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
  config.encoder.drop_path_rate = 0.0
  config.encoder.two_stream = 10

  config.encoder.g_mlp_dim = 4096
  config.encoder.g_num_heads = 16
  config.encoder.g_attention_dropout_rate = 0.0
  config.encoder.g_dropout_rate = 0.0
  config.encoder.g_predict_pos = False
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
  config.trainer = 'train_xlnet'
  # Where to search for pretrained ViT models.
  # Can be downloaded from gs://vit_models/imagenet21k
  # config.pretrained_dir = None
  # Which dataset to finetune on. This can be the name of a tfds dataset
  # (see https://www.tensorflow.org/datasets/catalog/overview), or the path to
  # a directory with the following structure ($filename can be arbitrary):
  # "{train,test}/$class_name/$filename.jpg"
  config.dataset = '/data/LargeData/Large/ImageNet/'
  # Path to manually downloaded dataset
  # config.tfds_manual_dir = None
  # Path to tensorflow_datasets directory
  # config.tfds_data_dir = None
  # Number of steps; determined by hyper module if not specified.
  config.mask_ratio = 0.75
  config.normlize_target = True

  config.weight_decay = 0.05
  config.base_lr = 1.5e-4
  config.decay_type = 'cosine'
  config.warmup_epochs = 40
  config.epochs = 1600

  config.beta1 = 0.9
  config.beta2 = 0.95

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
  config.flip = False
  config.randaug = None

  config.optim_half_precision = False
  config.dynamic_scale = True

  config.target_ratio = 0.25

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
  config.out_dim = 768
  config.sigma2 = None

  config.patch_size = config.model.encoder.patches['size'][0]
  config.num_patches = (config.pp['crop'] // config.model.encoder.patches['size'][0])**2
  config.num_mask = int(config.num_patches * config.mask_ratio)
  config.num_target = int(config.num_patches * config.target_ratio)

  return config.lock()
