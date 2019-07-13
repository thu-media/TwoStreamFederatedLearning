# -*- coding: utf-8 -*-
"""Print and merge checkpoint files of tensorflow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import mnist
import mnist_deep
import mnist_dataset

# Basic model parameters as external flags.
import external


def print_checkpoint_file(filename):
  # checkpoint_file = os.path.join(external.FLAGS.train_dir, filename)
  # print_tensors_in_checkpoint_file(checkpoint_file, tensor_name=None, all_tensors=True)

  checkpoint_file = os.path.join(external.FLAGS.train_dir, external.CKPT_FILE)
  try:
    reader = tf.train.NewCheckpointReader(checkpoint_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
      print("tensor_name: ", key)
      # print(type(reader.get_tensor(key)))
      # print(reader.get_tensor(key))
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))


def merge_checkpoint_file():
  dg = tf.Graph()
  with dg.as_default():
    x, y_ = mnist_dataset.placeholder_inputs(external.FLAGS.batch_size)
    # Build the graph for the deep net
    y_conv, _ = mnist_deep.deepnn(x)
    cross_entropy = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.to_int64(y_), logits=y_conv))

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()
    sess.run(init_op)
    # with tf.variable_scope('hidden1', reuse=True):
    #   weights = tf.get_variable('weights')
    #   biases = tf.get_variable('biases')
    # restore_saver = tf.train.Saver({'hidden1/weights': weights, 'hidden1/biases': biases})
    restore_saver = tf.train.Saver(var_list=tf.trainable_variables())
    multi_models = []
    for filename in external.FLAGS.model_list:
      checkpoint_file = os.path.join(external.FLAGS.train_dir, filename)
      restore_saver.restore(sess, checkpoint_file)
      # weights1 = weights.eval(session=sess)
      # print(weights)
      # print(type(weights1))
      # print(weights1)

      gc = tf.trainable_variables()
      # print(gc)
      model_now = []
      for var in gc:
        # print(var)
        var_v = var.eval(session=sess)
        # print(var_v)
        model_now.append(var_v)
        # print(model_now)
      multi_models.append(model_now)

    # assign_op = weights.assign((weights1 + weights2)/2)
    # sess.run(assign_op)
    # print(weights)
    # print(dg.get_tensor_by_name(name='hidden1/weights:0').eval(session=sess))
    # print(weights.eval(session=sess))

    multi_models_mean = np.mean(multi_models, axis=0)
    # print('Model mean:')
    # print(multi_models_mean)

    for i, _ in enumerate(gc):
      sess.run(gc[i].assign(multi_models_mean[i]))
    t = sess.run(gc[5])
    print(t)

    mean_ckpt_file = os.path.join(external.FLAGS.train_dir, external.FLAGS.merged_model)
    restore_saver.save(sess, mean_ckpt_file)
    print('Merged model saved.')
    sess.close()


def main(_):
  external.TRAIN_FILE = 'train.tfrecords'
  external.TEST_FILE = 'test.tfrecords'
  external.CKPT_FILE = 'deep_model_1m2.ckpt'

  merge_checkpoint_file()
  # print_checkpoint_file(_)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--merged_model',
    type=str,
    default='deep_model_1m2.ckpt',
    help='File to save merged model.'
  )
  parser.add_argument(
    '--model_list',
    type=str,
    nargs='+',
    default=['deep_model_1.ckpt', 'deep_model_2.ckpt'],
    help='Model files for merge .'
  )
  parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.01,
    help='Initial learning rate.'
  )
  parser.add_argument(
    '--num_epochs',
    type=int,
    default=2,
    help='Number of epochs to run trainer.'
  )
  parser.add_argument(
    '--batch_size',
    type=int,
    default=50,
    help='Batch size.'
  )
  parser.add_argument(
    '--train_dir',
    type=str,
    default='./input_data',
    help='Directory with the training data.'
  )

  external.FLAGS, unparsed = parser.parse_known_args()
  # print(external.FLAGS)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
