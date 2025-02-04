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

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf

from jax.lib import xla_bridge as xb
from jax.lib import xla_extension as xc

from vit_jax import inference_time
from vit_jax import train
from vit_jax import train_mae
from vit_jax import train_xlnet
from vit_jax import ft
from vit_jax import utils


FLAGS = flags.FLAGS

_WORKDIR = flags.DEFINE_string('workdir', None,
                               'Directory to store logs and model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)
flags.mark_flags_as_required(['config', 'workdir'])
flags.DEFINE_string('server_ip', '', help='IP of rank 0 server.')
flags.DEFINE_integer('server_port', 0, help='port of rank 0 server.')
flags.DEFINE_integer('num_hosts', 1, help='number of nodes in GPU cluster.')
flags.DEFINE_integer('host_idx', 0, help='index of current node.')
# Flags --jax_backend_target and --jax_xla_backend are available through JAX.

def _reset_backend_state():
  xb._backends = {}
  xb._backends_errors = {}
  xb._default_backend = None
  xb.get_backend.cache_clear()

def connect_to_gpu_cluster():
  _reset_backend_state()
  service = None
  if FLAGS.host_idx == 0:
    addr = f'{FLAGS.server_ip}:{FLAGS.server_port}'
    logging.info('starting service on %s', addr)
    service = xc.get_distributed_runtime_service(addr, FLAGS.num_hosts)
    # We add an explicit call to shutdown the service via atexit, since it seems
    # Python interpreter may not call the service destructor on process
    # termination.
    atexit.register(service.shutdown)

  server_addr = f'{FLAGS.server_ip}:{FLAGS.server_port}'
  logging.info('connecting to service on %s', server_addr)
  dist_client = xc.get_distributed_runtime_client(server_addr, FLAGS.host_idx)
  dist_client.connect()
  atexit.register(dist_client.shutdown)

  # register dist gpu backend
  factory = functools.partial(jax.lib.xla_client.make_gpu_client, dist_client, FLAGS.host_idx)
  xb.register_backend_factory('gpu', factory, priority=300)
  return service

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # each training node needs to connect to rank 0 server
  # service = connect_to_gpu_cluster()
  logging.info('gpu cluster connected with %d GPUs', jax.device_count())

  logging.info('devices %s', jax.devices())
  logging.info('local devices %s', jax.local_devices())

  utils.add_gfile_logger(_WORKDIR.value)

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  jax.config.update('jax_log_compiles', True)

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())
  jax_xla_backend = ('None' if FLAGS.jax_xla_backend is None else
                     FLAGS.jax_xla_backend)
  logging.info('Using JAX XLA backend %s', jax_xla_backend)

  logging.info('Config: %s', FLAGS.config)

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       _WORKDIR.value, 'workdir')
    
  if FLAGS.config.trainer == 'train':
    train.train_and_evaluate(FLAGS.config, _WORKDIR.value)
  elif FLAGS.config.trainer == 'inference_time':
    inference_time.inference_time(FLAGS.config, _WORKDIR.value)
  elif FLAGS.config.trainer == 'train_mae':
    train_mae.train_and_evaluate(FLAGS.config, _WORKDIR.value)
  elif FLAGS.config.trainer == 'train_xlnet':
    train_xlnet.train_and_evaluate(FLAGS.config, _WORKDIR.value)
  elif 'finetune' in FLAGS.config.trainer:
    ft.train_and_evaluate(FLAGS.config, _WORKDIR.value)
  elif FLAGS.config.trainer == 'inference_time':
    inference_time.inference_time(FLAGS.config, _WORKDIR.value)
  else:
    raise app.UsageError(f'Unknown trainer: {FLAGS.config.trainer}')

if __name__ == '__main__':
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  app.run(main)
