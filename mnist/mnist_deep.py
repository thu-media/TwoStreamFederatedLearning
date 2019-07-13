# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import mnist_dataset
import external
import losses

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def deepnn(x, trainable=True):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable('w_conv1', [5, 5, 1, 32], trainable=trainable)
  b_conv1 = bias_variable('b_conv1', [32], trainable=trainable)
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable('w_conv2', [5, 5, 32, 64], trainable=trainable)
  b_conv2 = bias_variable('b_conv2', [64], trainable=trainable)
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable('w_fc1', [7 * 7 * 64, 1024], trainable=trainable)
  b_fc1 = bias_variable('b_fc1', [1024], trainable=trainable)

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable('w_fc2', [1024, 10], trainable=trainable)
  b_fc2 = bias_variable('b_fc2', [10], trainable=trainable)
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(name, shape, trainable=True):
  """weight_variable generates a weight variable of a given shape."""
  # initial = tf.truncated_normal(shape, stddev=0.1)
  # return tf.Variable(initial)
  return tf.get_variable(name, shape, trainable=trainable, initializer=tf.truncated_normal_initializer(stddev=0.1))


def bias_variable(name, shape, trainable=True):
  """bias_variable generates a bias variable of a given shape."""
  # initial = tf.constant(0.1, shape=shape)
  # return tf.Variable(initial)
  return tf.get_variable(name, shape, trainable=trainable, initializer=tf.constant_initializer(0.1))


def do_eval():
  # reset the default graph
  tf.reset_default_graph()

  x, y_ = mnist_dataset.input(train=False, batch_size=external.FLAGS.batch_size, num_epochs=1)
  y_conv, keep_prob = deepnn(x)
  correct_prediction = tf.nn.in_top_k(y_conv, y_, 1)
  correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
  data_num = tf.shape(y_conv)[0]

  with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
    sess.run(init_op)

    # restore model
    saver = tf.train.Saver()
    checkpoint_file = os.path.join(external.FLAGS.train_dir, external.FLAGS.model_file)
    saver.restore(sess, checkpoint_file)
    print("Model Restored.")

    num_examples, true_count = 0, 0
    try:
      step = 0
      while True:
        feed_dict = {
          keep_prob: 1.0
        }
        temp_num, temp_correct = sess.run([data_num, correct_num], feed_dict=feed_dict)
        num_examples += temp_num
        true_count += temp_correct
        step += 1
    except tf.errors.OutOfRangeError:
      print('Done testing for %d epochs, %d steps.' % (1, step))

    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

    sess.close()


def train():
  # reset the default graph
  tf.reset_default_graph()

  # Import data
  x, y_ = mnist_dataset.input(train=True, batch_size=external.FLAGS.batch_size, num_epochs=external.FLAGS.num_epochs)

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  cross_entropy = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.to_int64(y_), logits=y_conv))
  correct_prediction = tf.nn.in_top_k(y_conv, y_, 1)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  data_num = tf.shape(y_conv)[0]
  train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

  # get the list of local model
  local_model = tf.trainable_variables()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # restore model
    saver = tf.train.Saver(var_list=local_model)
    checkpoint_file = os.path.join(external.FLAGS.train_dir, external.FLAGS.init_file)
    saver.restore(sess, checkpoint_file)
    print("Model Restored.")

    # get the list of global model
    # global_model = list()
    # for t in local_model:
    #   tensor_v = sess.run(t)
    #   global_model.append(tensor_v)

    # l2_reg = tf.reduce_sum([tf.nn.l2_loss(v - local_model[i]) for (i, v) in enumerate(global_model)])
    # l2_lambda = 0.1
    # total_loss = cross_entropy# + l2_lambda * l2_reg

    try:
      num_examples, step = 0, 0
      feed_dict = {keep_prob: 1.0}
      while True:
        if step % 100 == 0:
          train_accuracy, cross_entropy_v = sess.run([accuracy, cross_entropy], feed_dict=feed_dict)
          print('Step %d, training accuracy %g' % (step, train_accuracy))
          print('cross_entropy: ' + str(cross_entropy_v))
        else:
          _, temp = sess.run([train_step, data_num], feed_dict=feed_dict)
          num_examples += temp
        step += 1
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (external.FLAGS.num_epochs, step))
      print('Total records number: %d.' % num_examples)
      checkpoint_file = os.path.join(external.FLAGS.train_dir, external.FLAGS.model_file)
      saver.save(sess, checkpoint_file)

    sess.close()


def train_double():
  # reset the default graph
  tf.reset_default_graph()

  # Import data
  image, label = mnist_dataset.input(train=True, batch_size=external.FLAGS.batch_size, num_epochs=external.FLAGS.num_epochs)

  # global model
  with tf.variable_scope('global'):
    logits_1, keep_prob_1 = deepnn(image, trainable=False)
  # local model
  with tf.variable_scope('local'):
    logits_2, keep_prob_2 = deepnn(image, trainable=True)

  cross_entropy = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.to_int64(label), logits=logits_2))
  correct_prediction = tf.nn.in_top_k(logits_2, label, 1)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  data_num = tf.shape(image)[0]

  all_vars = tf.global_variables()
  half_len = len(all_vars) // 2
  global_model, local_model = all_vars[:half_len], all_vars[half_len:]
  global_var_list, local_var_list = dict(), dict()
  for i in range(half_len):
    name = all_vars[i].name.split('/')[1][:-2]
    global_var_list[name] = all_vars[i]
    local_var_list[name] = all_vars[i + half_len]
  # for k, v in local_var_list.items():
  #   print(k, v)

  # ldd = losses.mmd_loss(logits_1, logits_2, 0.1)
  ldd = 0.001 * tf.nn.l2_loss(logits_1 - logits_2)
  total_loss = cross_entropy + ldd
  train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(total_loss)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # restore model
    saver_global = tf.train.Saver(var_list=global_var_list)
    saver_local = tf.train.Saver(var_list=local_var_list)
    checkpoint_file = os.path.join(external.FLAGS.train_dir, external.FLAGS.init_file)
    saver_global.restore(sess, checkpoint_file)
    saver_local.restore(sess, checkpoint_file)
    print("Model Restored.")

    try:
      num_examples, step = 0, 0
      feed_dict = {
        keep_prob_1: 1.0,
        keep_prob_2: 1.0
      }
      while True:
        if step % 100 == 0:
          train_accuracy, cross_entropy_v, total_loss_v = sess.run([accuracy, cross_entropy, total_loss],
                                                                   feed_dict=feed_dict)
          print('Step %d, training accuracy %g' % (step, train_accuracy))
          print('cross_entropy: ' + str(cross_entropy_v))
          print('total_loss: ' + str(total_loss_v))
        else:
          _, temp = sess.run([train_step, data_num], feed_dict=feed_dict)
          num_examples += temp
        step += 1
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (external.FLAGS.num_epochs, step))
      print('Total records number: %d.' % num_examples)
      checkpoint_file = os.path.join(external.FLAGS.train_dir, external.FLAGS.model_file)
      saver_local.save(sess, checkpoint_file)

    sess.close()


def main(_):
  if external.FLAGS.test == 0:
    train()
  elif external.FLAGS.test == 1:
    do_eval()
  elif external.FLAGS.test == 2:
    train_double()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--train_file',
    type=str,
    nargs='+',
    default=['train.tfrecords'],
    help='Train filename.'
  )
  parser.add_argument(
    '--test_file',
    type=str,
    nargs='+',
    default=['test.tfrecords'],
    help='Test filename.'
  )
  parser.add_argument(
    '--model_file',
    type=str,
    default='deep_model_1m2.ckpt',
    help='File to save model.'
  )
  parser.add_argument(
    '--init_file',
    type=str,
    default='deep_model_init.ckpt',
    help='File to restore model.'
  )
  parser.add_argument(
    '--test',
    type=int,
    default=1,
    help='1 for test, 0 for train, 2 for train double.'
  )
  parser.add_argument(
    '--train_dir',
    type=str,
    default='./input_data/',
    help='Directory for storing input data'
  )
  parser.add_argument(
    '--batch_size',
    type=int,
    default=50,
    help='Batch size.'
  )
  parser.add_argument(
    '--num_epochs',
    type=int,
    default=2,
    help='Num of epoch.'
  )
  external.FLAGS, unparsed = parser.parse_known_args()
  # print(external.FLAGS)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
