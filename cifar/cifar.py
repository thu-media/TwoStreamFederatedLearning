# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import os

import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3


class Cifar10DataSet(object):
  """Cifar10 data set.

  Described by http://www.cs.toronto.edu/~kriz/cifar.html.
  """

  def __init__(self, data_dir, subset='train', use_distortion=True):
    self.data_dir = data_dir
    self.subset = subset
    self.use_distortion = use_distortion

  def get_filenames(self, file_nums=None):
    '''file_nums: the list of file numbers for reading'''
    if self.subset in ['train', 'validation', 'eval']:
      if not file_nums:
        return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
      else:
        return [os.path.join(self.data_dir, self.subset + '_%d.tfrecords' % i) for i in file_nums]
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def preprocess(self, image):
    """Preprocess a single image in [height, width, depth] layout."""
    if self.subset == 'train' and self.use_distortion:
      # Pad 4 pixels on each dimension of feature map, done in mini-batch
      image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
      image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
      image = tf.image.random_flip_left_right(image)
      image = tf.image.random_brightness(image, max_delta=63)
      image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [height, width, depth].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)

    # Custom preprocessing.
    image = self.preprocess(image)

    return image, label

  def make_batch(self, batch_size, num_epochs=None, file_nums=None):
    '''file_nums: the list of file numbers for reading'''
    filenames = self.get_filenames(file_nums)
    # Repeat num_epochs times, None for infinitely.
    dataset = tf.data.TFRecordDataset(filenames).repeat(num_epochs)

    # Parse records.
    dataset = dataset.map(self.parser, num_parallel_calls=6)

    # Potentially shuffle records.
    if self.subset == 'train':
      min_queue_examples = int(
          Cifar10DataSet.num_examples_per_epoch(self.subset) * 0.4)
      # Ensure that the capacity is sufficiently large to provide good random
      # shuffling.
      dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    # Batch it up.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

  @staticmethod
  def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 45000
    elif subset == 'validation':
      return 5000
    elif subset == 'eval':
      return 10000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)


def main():
  dataset = Cifar10DataSet('./input_data')
  image_batch, label_batch = dataset.make_batch(10, file_nums=[0, 1])
  with tf.Session() as sess:
    image_shape, label_shape = sess.run([tf.shape(image_batch), tf.shape(label_batch)])
    print(image_shape, label_shape)  #[10 32 32 3] [10]


if __name__ == '__main__':
  main()
