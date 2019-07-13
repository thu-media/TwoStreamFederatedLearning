# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def print_activations(t):
  # print(t.op.name, ' ', t.get_shape().as_list())
  pass


def inference(images, trainable=True, use_dropout=False, last=True):
  """Build the AlexNet model for cifar10.

  Args:
    images: Images Tensor
    use_dropout: whether to use dropout.

  Returns:
    logits: the last Tensor in the softmax layer of AlexNet, in the shape [batch, 10]

  Data format: (batch, height, width, channels)

  Conv1: kernel:  5x5, strides: 1, out channel: 64
  Pool1: pool size: 3, strides: 2

  Conv2: kenerl:  5x5, strides: 1, out channel: 128
  Pool2: pool size: 3, strides: 2

  Conv3: kernel:  3x3, strides: 1, out channel: 256

  Conv4: kernel:  3x3, strides: 1, out channel: 256

  Conv5: kernel:  3x3, strides: 1, out channel: 128
  Pool5: pool size: 3, strides: 2

  Fc6:   output dimensions: 1024

  Fc7:   output dimensions: 1024
  """
  # conv1
  with tf.name_scope('conv1') as scope:
    conv1 = tf.layers.conv2d(
      inputs=images,
      filters=64,
      kernel_size=[5, 5],
      strides=[1, 1],
      padding='SAME',
      kernel_initializer=tf.glorot_uniform_initializer(),
      trainable=trainable,
      activation=tf.nn.relu)
    print_activations(conv1)

  # pool1
  with tf.name_scope('pool1') as scope:
    pool1 = tf.layers.max_pooling2d(
      inputs=conv1,
      pool_size=[3, 3],
      strides=2,
      padding='VALID')
    print_activations(pool1)

  # conv2
  with tf.name_scope('conv2') as scope:
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[5, 5],
      strides=[1, 1],
      padding='SAME',
      kernel_initializer=tf.glorot_uniform_initializer(),
      trainable=trainable,
      activation=tf.nn.relu)
    print_activations(conv2)

  # pool2
  with tf.name_scope('pool2') as scope:
    pool2 = tf.layers.max_pooling2d(
      inputs=conv2,
      pool_size=[3, 3],
      strides=2,
      padding='VALID')
    print_activations(pool2)

  # conv3
  with tf.name_scope('conv3') as scope:
    conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=256,
      kernel_size=[3, 3],
      strides=[1, 1],
      padding='SAME',
      kernel_initializer=tf.glorot_uniform_initializer(),
      trainable=trainable,
      activation=tf.nn.relu)
    print_activations(conv3)

  # conv4
  with tf.name_scope('conv4') as scope:
    conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=256,
      kernel_size=[3, 3],
      strides=[1, 1],
      padding='SAME',
      kernel_initializer=tf.glorot_uniform_initializer(),
      trainable=trainable,
      activation=tf.nn.relu)
    print_activations(conv4)

  # conv5
  with tf.name_scope('conv5') as scope:
    conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=128,
      kernel_size=[3, 3],
      strides=[1, 1],
      padding='SAME',
      kernel_initializer=tf.glorot_uniform_initializer(),
      trainable=trainable,
      activation=tf.nn.relu)
    print_activations(conv5)

  # pool5
  with tf.name_scope('pool5') as scope:
    pool5 = tf.layers.max_pooling2d(
      inputs=conv5,
      pool_size=[3, 3],
      strides=2,
      padding='VALID')
    print_activations(pool5)
    pool5_flat = tf.layers.flatten(inputs=pool5)

  # fc6
  with tf.name_scope('fc6') as scope:
    fc6 = tf.layers.dense(
      inputs=pool5_flat,
      units=1024,
      kernel_initializer=tf.glorot_uniform_initializer(),
      trainable=trainable,
      activation=tf.nn.relu)
    fc6_dropout = tf.layers.dropout(
      inputs=fc6,
      rate=0.1,
      training=use_dropout)

  # fc7
  with tf.name_scope('fc7') as scope:
    fc7 = tf.layers.dense(
      inputs=fc6_dropout,
      units=1024,
      kernel_initializer=tf.glorot_uniform_initializer(),
      trainable=trainable,
      activation=tf.nn.relu)
    fc7_dropout = tf.layers.dropout(
      inputs=fc7,
      rate=0.1,
      training=use_dropout)

  # softmax layer
  with tf.name_scope('softmax') as scope:
    logits = tf.layers.dense(
      inputs=fc7_dropout,
      units=10,
      kernel_initializer=tf.glorot_uniform_initializer(),
      trainable=last)

  return logits, fc7
