# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import time
import math
import argparse
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

# Basic model parameters as external flags.
import external

import mnist_dataset

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def inference(images, hidden1_units, hidden2_units, mask=None):
  """Build the MNIST model up to where it may be used for inference.
  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
  Returns:
    softmax_linear: Output tensor with the computed logits.
    hidden2: Output tensor of 'hidden2' layer.
  """
  # Hidden 1
  with tf.variable_scope('hidden1'):
    weights = tf.get_variable('weights', [IMAGE_PIXELS, hidden1_units],
        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))))
    biases = tf.get_variable('biases', [hidden1_units],
        initializer=tf.constant_initializer(0.0))
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.variable_scope('hidden2'):
    weights = tf.get_variable('weights', [hidden1_units, hidden2_units],
        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(hidden1_units))))
    biases = tf.get_variable('biases', [hidden2_units],
        initializer=tf.constant_initializer(0.0))
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    if mask:
      hidden2 = tf.multiply(hidden2, mask)
  # Linear
  with tf.variable_scope('softmax_linear'):
    weights = tf.get_variable('weights', [hidden2_units, NUM_CLASSES],
        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(hidden2_units))))
    biases = tf.get_variable('biases', [NUM_CLASSES],
        initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(hidden2, weights) + biases

  return logits, hidden2


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  # global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))


def do_eval():
  # Tell TensorFlow that the model will be built into the default Graph.
  tf.reset_default_graph()
  with tf.Graph().as_default():
    images_placeholder, labels_placeholder = mnist_dataset.placeholder_inputs(external.FLAGS.batch_size)
    # Input images and labels.
    images, labels, reader = mnist_dataset.inputs(train=False, batch_size=external.FLAGS.batch_size,
                            num_epochs=external.FLAGS.num_epochs)
    # Build a Graph that computes predictions from the inference model.
    logits, _ = inference(images_placeholder,
                             external.FLAGS.hidden1,
                             external.FLAGS.hidden2)

    eval_correct = evaluation(logits, labels_placeholder)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running operations in the Graph.
    sess = tf.Session()

    # Initialize the variables (the trained variables and the
    # epoch counter).
    sess.run(init_op)

    checkpoint_file = os.path.join(external.FLAGS.train_dir, external.CKPT_FILE)
    saver.restore(sess, checkpoint_file)
    print("Model restored.")

    true_count = 0
    try:
      step = 0
      while True:
        test_images, test_labels = sess.run([images, labels])
        feed_dict = {
          images_placeholder: test_images,
          labels_placeholder: test_labels,
        }
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
        step += 1
    except tf.errors.OutOfRangeError:
      print('Done testing for %d epochs, %d steps.' % (external.FLAGS.num_epochs, step))

    num_examples = sess.run(reader.num_records_produced())
    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

    sess.close()


def run_training():
  """Train MNIST for a number of steps."""
  tf.reset_default_graph()
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    images_placeholder, labels_placeholder = mnist_dataset.placeholder_inputs(external.FLAGS.batch_size)
    # Input images and labels.
    images, labels, reader = mnist_dataset.inputs(train=True, batch_size=external.FLAGS.batch_size,
                            num_epochs=external.FLAGS.num_epochs)
    # Build a Graph that computes predictions from the inference model.
    logits, _ = inference(images_placeholder,
                             external.FLAGS.hidden1,
                             external.FLAGS.hidden2)

    # Add to the Graph the loss calculation.
    train_loss = loss(logits, labels_placeholder)

    # Add to the Graph operations that train the model.
    train_op = training(train_loss, external.FLAGS.learning_rate)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Create a session for running operations in the Graph.
    sess = tf.Session()

    # Initialize the variables (the trained variables and the
    # epoch counter).
    sess.run(init_op)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    init_ckpt = os.path.join(external.FLAGS.train_dir, 'model_1m2.ckpt')
    # print(init_ckpt)
    saver.restore(sess, init_ckpt)

    try:
      step = 0
      while True:
        start_time = time.time()
        # Run one step of the model.  The return values are
        # the activations from the `train_op` (which is
        # discarded) and the `loss` op.  To inspect the values
        # of your ops or variables, you may include them in
        # the list passed to sess.run() and the value tensors
        # will be returned in the tuple from the call.
        train_images, train_labels = sess.run([images, labels])
        feed_dict = {
          images_placeholder: train_images,
          labels_placeholder: train_labels,
        }
        _, loss_value = sess.run([train_op, train_loss], feed_dict=feed_dict)

        duration = time.time() - start_time

        # Print an overview fairly often.
        if step % 100 == 0:
          print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        step += 1
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (external.FLAGS.num_epochs, step))
      print('Total records number: %d.' % sess.run(reader.num_records_produced()))
      checkpoint_file = os.path.join(external.FLAGS.train_dir, external.CKPT_FILE)
      saver.save(sess, checkpoint_file)

    sess.close()


def main(_):
  external.TRAIN_FILE = external.FLAGS.train_file
  external.TEST_FILE = external.FLAGS.test_file
  external.CKPT_FILE = external.FLAGS.model_file
  if not external.FLAGS.test:
    run_training()
  do_eval()


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
    default='model.ckpt',
    help='Checkpoint filename.'
  )
  parser.add_argument(
    '--test',
    type=int,
    default=1,
    help='1 for test, 0 for train.'
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
    '--hidden1',
    type=int,
    default=128,
    help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
    '--hidden2',
    type=int,
    default=32,
    help='Number of units in hidden layer 2.'
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
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)