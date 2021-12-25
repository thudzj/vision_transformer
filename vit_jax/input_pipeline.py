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

import glob
import os

from absl import logging
import flax
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import sys
if sys.platform != 'darwin':
  # A workaround to avoid crash because tfds may open to many files.
  import resource
  low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# Adjust depending on the available RAM.
_RESIZE_MIN = 256
MAX_IN_MEMORY = 200_000
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

def normalize_image(image):
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image

def _central_crop(image, crop_height, crop_width):
  """Performs central crops of the given image list.
  Args:
    image: a 3-D image tensor
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.
  Returns:
    3-D tensor with cropped image.
  """
  shape = tf.shape(image)
  height, width = shape[0], shape[1]

  amount_to_be_cropped_h = (height - crop_height)
  crop_top = amount_to_be_cropped_h // 2
  amount_to_be_cropped_w = (width - crop_width)
  crop_left = amount_to_be_cropped_w // 2
  return tf.slice(
      image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def _smallest_size_at_least(height, width, resize_min):
  """Computes new shape with the smallest side equal to `smallest_side`.
  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.
  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: an int32 scalar tensor indicating the new width.
  """
  resize_min = tf.cast(resize_min, tf.float32)

  # Convert to floats to make subsequent calculations go smoothly.
  height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

  smaller_dim = tf.minimum(height, width)
  scale_ratio = resize_min / smaller_dim

  # Convert back to ints to make heights and widths that TF ops will accept.
  new_height = tf.cast(height * scale_ratio, tf.int32)
  new_width = tf.cast(width * scale_ratio, tf.int32)

  return new_height, new_width

def _aspect_preserving_resize(image, resize_min):
  """Resize images preserving the original aspect ratio.
  Args:
    image: A 3-D image `Tensor`.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  shape = tf.shape(image)
  height, width = shape[0], shape[1]

  new_height, new_width = _smallest_size_at_least(height, width, resize_min)

  return _resize_image(image, new_height, new_width)


def _resize_image(image, height, width):
  """Simple wrapper around tf.resize_images.
  This is primarily to make sure we use the same `ResizeMethod` and other
  details each time.
  Args:
    image: A 3-D image `Tensor`.
    height: The target height for the resized image.
    width: The target width for the resized image.
  Returns:
    resized_image: A 3-D tensor containing the resized image. The first two
      dimensions have the shape [height, width].
  """
  return tf.image.resize(
      image, [height, width], method=tf.image.ResizeMethod.BILINEAR)

def get_tfds_info(dataset, split):
  """Returns information about tfds dataset -- see `get_dataset_info()`."""
  data_builder = tfds.builder(dataset)
  return dict(
      num_examples=data_builder.info.splits[split].num_examples,
      num_classes=data_builder.info.features['label'].num_classes,
      int2str=data_builder.info.features['label'].int2str,
      examples_glob=None,
  )


def get_directory_info(directory):
  """Returns information about directory dataset -- see `get_dataset_info()`."""
  examples_glob = f'{directory}/*/*.JPEG'
  paths = glob.glob(examples_glob)
  get_classname = lambda path: path.split('/')[-2]
  class_names = sorted(set(map(get_classname, paths)))
  return dict(
      num_examples=len(paths),
      num_classes=len(class_names),
      int2str=lambda id_: class_names[id_],
      examples_glob=examples_glob,
  )


def get_dataset_info(dataset, split):
  """Returns information about a dataset.
  
  Args:
    dataset: Name of tfds dataset or directory -- see `./configs/common.py`
    split: Which split to return data for (e.g. "test", or "train"; tfds also
      supports splits like "test[:90%]").

  Returns:
    A dictionary with the following keys:
    - num_examples: Number of examples in dataset/mode.
    - num_classes: Number of classes in dataset.
    - int2str: Function converting class id to class name.
    - examples_glob: Glob to select all files, or None (for tfds dataset).
  """
  directory = os.path.join(dataset, split)
  if os.path.isdir(directory):
    return get_directory_info(directory)
  return get_tfds_info(dataset, split)


def get_datasets(config):
  """Returns `ds_train, ds_test` for specified `config`."""

  if os.path.isdir(config.dataset):
    train_dir = os.path.join(config.dataset, config.pp['train'])
    test_dir = os.path.join(config.dataset, config.pp['test'])
    if not os.path.isdir(train_dir):
      raise ValueError('Expected to find directories"{}" and "{}"'.format(
          train_dir,
          test_dir,
      ))
    logging.info('Reading dataset from directories "%s" and "%s"', train_dir,
                 test_dir)
    ds_train = get_data_from_directory(
        config=config, directory=train_dir, mode='train')
    ds_test = get_data_from_directory(
        config=config, directory=test_dir, mode='test')
  else:
    logging.info('Reading dataset from tfds "%s"', config.dataset)
    ds_train = get_data_from_tfds(config=config, mode='train')
    ds_test = get_data_from_tfds(config=config, mode='test')

  return ds_train, ds_test


def get_data_from_directory(*, config, directory, mode):
  """Returns dataset as read from specified `directory`."""

  dataset_info = get_directory_info(directory)
  data = tf.data.Dataset.list_files(dataset_info['examples_glob'])
  class_names = [
      dataset_info['int2str'](id_) for id_ in range(dataset_info['num_classes'])
  ]

  def _pp(path):
    return dict(
        image=path,
        label=tf.where(
            tf.strings.split(path, '/')[-2] == class_names
        )[0][0],
    )

  image_decoder = lambda path: tf.image.decode_jpeg(tf.io.read_file(path), 3)

  if config.trainer == 'train_mae':
    return_mask = True
    num_patches = config.num_patches
  else:
    return_mask = False
    num_patches = None

  return get_data(
      data=data,
      mode=mode,
      num_classes=dataset_info['num_classes'],
      image_decoder=image_decoder,
      repeats=None if mode == 'train' else 1,
      batch_size=config.batch_eval if mode == 'test' else config.batch,
      image_size=config.pp['crop'],
      shuffle_buffer=min(dataset_info['num_examples'], config.shuffle_buffer),
      preprocess=_pp,
      return_mask=return_mask,
      num_patches=num_patches)


def get_data_from_tfds(*, config, mode):
  """Returns dataset as read from tfds dataset `config.dataset`."""

  data_builder = tfds.builder(config.dataset, data_dir=config.tfds_data_dir)

  data_builder.download_and_prepare(
      download_config=tfds.download.DownloadConfig(
          manual_dir=config.tfds_manual_dir))
  data = data_builder.as_dataset(
      split=config.pp[mode],
      # Reduces memory footprint in shuffle buffer.
      decoders={'image': tfds.decode.SkipDecoding()},
      shuffle_files=mode == 'train')
  image_decoder = data_builder.info.features['image'].decode_example

  dataset_info = get_tfds_info(config.dataset, config.pp[mode])
  return get_data(
      data=data,
      mode=mode,
      num_classes=dataset_info['num_classes'],
      image_decoder=image_decoder,
      repeats=None if mode == 'train' else 1,
      batch_size=config.batch_eval if mode == 'test' else config.batch,
      image_size=config.pp['crop'],
      shuffle_buffer=min(dataset_info['num_examples'], config.shuffle_buffer))



def get_data(*,
             data,
             mode,
             num_classes,
             image_decoder,
             repeats,
             batch_size,
             image_size,
             shuffle_buffer,
             preprocess=None,
             return_mask=False,
             num_patches=None):
  """Returns dataset for training/eval.

  Args:
    data: tf.data.Dataset to read data from.
    mode: Must be "train" or "test".
    num_classes: Number of classes (used for one-hot encoding).
    image_decoder: Applied to `features['image']` after shuffling. Decoding the
      image after shuffling allows for a larger shuffle buffer.
    repeats: How many times the dataset should be repeated. For indefinite
      repeats specify None.
    batch_size: Global batch size. Note that the returned dataset will have
      dimensions [local_devices, batch_size / local_devices, ...].
    image_size: Image size after cropping (for training) / resizing (for
      evaluation).
    shuffle_buffer: Number of elements to preload the shuffle buffer with.
    preprocess: Optional preprocess function. This function will be applied to
      the dataset just after repeat/shuffling, and before the data augmentation
      preprocess step is applied.
  """

  def _pp(data):
    im = image_decoder(data['image'])
    if im.shape[-1] == 1:
      im = tf.repeat(im, 3, axis=-1)
    if mode == 'train':
      channels = im.shape[-1]
      begin, size, _ = tf.image.sample_distorted_bounding_box(
          tf.shape(im),
          tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
          area_range=(0.08, 1.0),
          min_object_covered=0.1,
          use_image_if_no_bounding_boxes=True)
      im = tf.slice(im, begin, size)
      # Unfortunately, the above operation loses the depth-dimension. So we
      # need to restore it the manual way.
      im.set_shape([None, None, channels])
      im = _resize_image(im, image_size, image_size)
      # if tf.random.uniform(shape=[]) > 0.5:
      #   im = tf.image.flip_left_right(im)
    else:
      im = _aspect_preserving_resize(im, _RESIZE_MIN)
      im = _central_crop(im, image_size, image_size)
    im = normalize_image(im)
    if return_mask:
      label = tf.random.shuffle(tf.range(num_patches))
      # np.random.permutation().astype(int) # [196]
    else:
      label = tf.one_hot(data['label'], num_classes)  # pylint: disable=no-value-for-parameter
    return {'image': im, 'label': label}

  data = data.repeat(repeats)
  if mode == 'train':
    data = data.shuffle(shuffle_buffer)
  if preprocess is not None:
    data = data.map(preprocess, tf.data.experimental.AUTOTUNE)
  data = data.map(_pp, tf.data.experimental.AUTOTUNE)
  data = data.batch(batch_size, drop_remainder=True)

  # Shard data such that it can be distributed accross devices
  num_devices = jax.local_device_count()

  def _shard(data):
    data['image'] = tf.reshape(data['image'],
                               [num_devices, -1, image_size, image_size, 3])
    if return_mask:
      data['label'] = tf.reshape(data['label'],
                                 [num_devices, -1, num_patches])
    else:
      data['label'] = tf.reshape(data['label'],
                                 [num_devices, -1, num_classes])
    return data

  if num_devices is not None:
    data = data.map(_shard, tf.data.experimental.AUTOTUNE)

  return data.prefetch(1)


def prefetch(dataset, n_prefetch):
  """Prefetches data to device and converts to numpy array."""
  ds_iter = iter(dataset)
  ds_iter = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x),
                ds_iter)
  if n_prefetch:
    ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
  return ds_iter
