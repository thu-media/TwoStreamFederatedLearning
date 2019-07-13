# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import external

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  
  return images_placeholder, labels_placeholder


def input(train, batch_size, num_epochs):
  '''
  Dataset wrap for mnist.
  '''
  if not num_epochs: num_epochs = None
  filenames = []
  for filename in external.FLAGS.train_file if train else external.FLAGS.test_file:
    filenames.append(os.path.join(external.FLAGS.train_dir, filename))

  def _parse(example_proto):
    features = tf.parse_single_example(
      example_proto,
      features = {'image_raw': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64)}
      )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([IMAGE_PIXELS])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return image, label

  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_parse)  # Parse the record into tensors.
  if train:
    dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  next_image, next_label = iterator.get_next()

  return next_image, next_label


def inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  filenames = []
  for filename in external.TRAIN_FILE if train else external.TEST_FILE:
    filenames.append(os.path.join(external.FLAGS.train_dir, filename))

  def read_and_decode(filename_queue):
    '''
    Read and decode the examples from tfrecord files. 
    '''
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([IMAGE_PIXELS])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label, reader

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label, reader = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    if train:
      images, labels = tf.train.shuffle_batch(
          [image, label], batch_size=batch_size, num_threads=2,
          capacity=1000 + 3 * batch_size,
          # Ensures a minimum amount of shuffling of examples.
          min_after_dequeue=1000)
    else:
      images, labels = tf.train.batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size)

    return images, labels, reader