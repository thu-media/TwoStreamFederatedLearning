# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import numpy as np
import tensorflow as tf

import cifar
import cifar_model
import losses

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

FLAGS = None

def merge():
  dataset = cifar.Cifar10DataSet(FLAGS.data_dir, subset='eval', use_distortion=False)
  image_batch, _ = dataset.make_batch(FLAGS.batch_size, num_epochs=1)
  _, _ = cifar_model.inference(image_batch, use_dropout=False)
  # cross_entropy = tf.reduce_mean(
  #     tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.to_int64(label_batch), logits=logits))
  # train_step = tf.train.GradientDescentOptimizer(0).minimize(cross_entropy)

  var_list = tf.global_variables()
  # for t in var_list:
  #   print(t.name)

  # get the values of models to be merged
  multi_models = list()
  try:
    for filename in FLAGS.model_list:
      checkpoint_file = os.path.join(FLAGS.model_dir, filename)
      reader = tf.train.NewCheckpointReader(checkpoint_file)
      model_temp = list()
      for key in var_list:
        # print(key.name[:-2])
        model_temp.append(reader.get_tensor(key.name[:-2]))
      multi_models.append(model_temp)
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))

  # average the values
  model_means = np.mean(multi_models, axis=0)

  with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    for i, v in enumerate(model_means):
      sess.run(var_list[i].assign(v))

    # save the averaged model
    saver = tf.train.Saver(var_list=var_list)
    output_file = os.path.join(FLAGS.model_dir, FLAGS.model_file)
    saver.save(sess, output_file)

    print('Models merged and saved.')
    sess.close()


def eval():
  dataset = cifar.Cifar10DataSet(FLAGS.data_dir, subset='eval', use_distortion=False)
  image_batch, label_batch = dataset.make_batch(FLAGS.batch_size, num_epochs=1, file_nums=FLAGS.eval_file)
  logits, _ = cifar_model.inference(image_batch, use_dropout=False)
  correct_prediction = tf.nn.in_top_k(logits, label_batch, 1)
  correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
  data_num = tf.shape(logits)[0]

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    saver = tf.train.Saver(var_list=tf.global_variables())
    checkpoint_file = os.path.join(FLAGS.model_dir, FLAGS.model_file)
    saver.restore(sess, checkpoint_file)

    try:
      num_examples, num_correct, step = 0, 0, 0
      while True:
        temp_correct, temp_example = sess.run([correct_num, data_num])
        num_correct += temp_correct
        num_examples += temp_example
        step += 1
    except tf.errors.OutOfRangeError:
      print('Eval for %d steps, %d examples, accuracy %f.' % (step, num_examples, float(num_correct) / num_examples))

  sess.close()


def train():
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
  dataset = cifar.Cifar10DataSet(FLAGS.data_dir, subset='train', use_distortion=True)
  image_batch, label_batch = dataset.make_batch(FLAGS.batch_size, num_epochs=FLAGS.num_epochs, file_nums=FLAGS.train_file)
  logits, _ = cifar_model.inference(image_batch, use_dropout=True)

  # learning rate: adam 1e-4
  if not FLAGS.lr:
    # FLAGS.lr = tf.train.exponential_decay(2e-3, global_step, 30000, 0.5, staircase=True)
    FLAGS.lr = 2e-3
    # tf.summary.scalar('learning_rate', FLAGS.lr)

  # define the loss
  cross_entropy = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.to_int64(label_batch), logits=logits))
  correct_prediction = tf.nn.in_top_k(logits, label_batch, 1)
  train_step = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(cross_entropy)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  data_num = tf.shape(logits)[0]
  tf.summary.scalar('accuracy', accuracy)
  tf.summary.scalar('loss', cross_entropy)

  var_list = tf.global_variables()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # restore all layers except the classification layer
    saver = tf.train.Saver(var_list=var_list)
    checkpoint_file = os.path.join(FLAGS.model_dir, FLAGS.init_file)
    saver.restore(sess, checkpoint_file)

    merged = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
    try:
      num_examples, step = 0, 0
      while True:
        # if step % 1000 == 0:
        #   saver.save(sess, checkpoint_file, global_step=step)
        if step % 100 == 0:
          cross_entropy_v, train_accuracy, summary = sess.run([cross_entropy, accuracy, merged])
          print('cross_entropy: ' + str(cross_entropy_v))
          # train_writer.add_summary(summary, step)
          print('Step %d, training accuracy %g' % (step, train_accuracy))
        else:
          _, temp, summary = sess.run([train_step, data_num, merged])
          # train_writer.add_summary(summary, step)
          num_examples += temp
        step += 1
    except tf.errors.OutOfRangeError:
      checkpoint_file = os.path.join(FLAGS.model_dir, FLAGS.model_file)
      saver.save(sess, checkpoint_file)
      print('Train for %d steps, %d examples.' % (step, num_examples))

  sess.close()


def train_double():
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
  dataset = cifar.Cifar10DataSet(FLAGS.data_dir, subset='train', use_distortion=True)
  image_batch, label_batch = dataset.make_batch(FLAGS.batch_size, num_epochs=FLAGS.num_epochs, file_nums=FLAGS.train_file)

  # if FLAGS.rd % 2 == 0:
  #   last = False
  # else:
  #   last = True

  logits_1, fc7_1 = cifar_model.inference(image_batch, trainable=False, use_dropout=False, last=False)
  logits_2, fc7_2 = cifar_model.inference(image_batch, trainable=True, use_dropout=False, last=True)

  all_vars = tf.global_variables()
  model_vars = list()
  for v in all_vars:
    if not v.name.startswith('batch'):
      model_vars.append(v)

  half_length = len(model_vars) // 2
  model_1, model_2 = model_vars[0:half_length], model_vars[half_length:half_length * 2]

  # for v in model_vars:
  #   print(v)
  # return
  half_length = len(all_vars) // 2
  var_list_1, var_list_2 = dict(), dict()
  for i in range(half_length):
    var_list_1[all_vars[i].name[:-2]] = all_vars[i]
    var_list_2[all_vars[i].name[:-2]] = all_vars[i + half_length]

  # for k, v in var_list_1.items():
  #   print(k, v)
  # for k, v in var_list_2.items():
  #   print(k, v)
  # return

  # define the loss
  cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.to_int64(label_batch), logits=logits_2))
  l2_loss = tf.reduce_sum([tf.nn.l2_loss(v1 - v2) for v1, v2 in zip(model_1[14:16], model_2[14:16])])

  lambda_1 = tf.placeholder(dtype=tf.float32, shape=())
  # ldd = losses.mmd_loss(logits_1, logits_2, 1)
  ldd = tf.nn.l2_loss(logits_1 - logits_2)
  total_loss = cross_entropy + lambda_1 * ldd# + 2e-5 * np.power(FLAGS.rd, 2) * l2_loss

  if not FLAGS.lr:
    FLAGS.lr = 2e-3
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_step = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(total_loss)
  correct_prediction = tf.nn.in_top_k(logits_2, label_batch, 1)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  data_num = tf.shape(logits_2)[0]

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    saver_1 = tf.train.Saver(var_list=var_list_1)
    saver_2 = tf.train.Saver(var_list=var_list_2)
    checkpoint_file = os.path.join(FLAGS.model_dir, FLAGS.init_file)
    saver_1.restore(sess, checkpoint_file)
    saver_2.restore(sess, checkpoint_file)
    # print(sess.run(model_vars[0][0][0][0]))
    # print(sess.run(model_vars[half_length][0][0][0]))
    # return

    try:
      num_examples, step = 0, 0
      lambda_v = 0.01
      while True:
        if step % 100 == 0:
          cross_entropy_v, total_loss_v, train_accuracy = sess.run([cross_entropy, total_loss, accuracy],
                                                                  feed_dict={lambda_1: lambda_v})
          print('cross_entropy: ' + str(cross_entropy_v))
          # print('lambda_v: ' + str(lambda_v))
          print('total_loss: ' + str(total_loss_v))
          print('Step %d, training accuracy %g' % (step, train_accuracy))
        else:
          _, cross_entropy_v, ldd_v, temp_num = sess.run([train_step, cross_entropy, ldd, data_num],
                                                        feed_dict={lambda_1: lambda_v})
          # if ldd_v != 0.0:
          #   # 1.0e3 for CAL
          #   lambda_v = min(cross_entropy_v / ldd_v, 0.1)
          # else:
          #   lambda_v = 0.0
          # print('cross_entropy: ' + str(cross_entropy_v))
          # print('ldd: ' + str(ldd_v))
          # print('lambda: ' + str(lambda_v) + '\n')
          num_examples += temp_num
        step += 1
    except tf.errors.OutOfRangeError:
      checkpoint_file = os.path.join(FLAGS.model_dir, FLAGS.model_file)
      saver_2.save(sess, checkpoint_file)
      print('Train for %d steps, %d examples.' % (step, num_examples))

  sess.close()


def stepwise_train():
  global_step = tf.train.get_or_create_global_step()

  dataset = cifar.Cifar10DataSet(FLAGS.data_dir, subset='train', use_distortion=True)
  image_batch, label_batch = dataset.make_batch(FLAGS.batch_size, num_epochs=FLAGS.num_epochs, file_nums=FLAGS.train_file)
  logits, _ = cifar_model.inference(image_batch, use_dropout=True)

  # learning rate
  if not FLAGS.lr:
    FLAGS.lr = tf.train.exponential_decay(1e-4, global_step, 20000, 0.2, staircase=True)

  # get the list of local model.
  # len: 30
  var_list = tf.trainable_variables()
  # model vars, bn vars
  local_model, bn_vars = list(), list()
  for v in tf.trainable_variables():
    if not v.name.startswith('batch'):
      local_model.append(v)
    else:
      bn_vars.append(v)
  first_step_vars = local_model[10:] + bn_vars
  second_step_vars = local_model[6:] + bn_vars
  third_step_vars = local_model + bn_vars
  # for t in second_step:
  #   print(t)
  # return

  # define the loss
  cross_entropy = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.to_int64(label_batch), logits=logits))
  first_train_step = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(cross_entropy, global_step=global_step, var_list=first_step_vars)
  second_train_step = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(cross_entropy, global_step=global_step, var_list=second_step_vars)
  third_train_step = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(cross_entropy, global_step=global_step, var_list=third_step_vars)
  train_step = (first_train_step, second_train_step, third_train_step)
  correct_prediction = tf.nn.in_top_k(logits, label_batch, 1)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  data_num = tf.shape(logits)[0]

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    init_op = tf.group(tf.global_variables_initializer(),
                  tf.local_variables_initializer())
    sess.run(init_op)
    # restore all layers except the classification layer
    saver = tf.train.Saver(var_list=var_list)
    checkpoint_file = os.path.join(FLAGS.model_dir, FLAGS.init_file)
    saver.restore(sess, checkpoint_file)

    try:
      num_examples, step = 0, 0
      cgpt = int(len(FLAGS.train_file) * 4500 / FLAGS.batch_size)
      while True:
        # if step % 1000 == 0:
        #   saver.save(sess, checkpoint_file, global_step=step)
        if step % 100 == 0:
          train_accuracy = sess.run(accuracy)
          print('Step %d, training accuracy %g' % (step, train_accuracy))
        else:
          _, temp = sess.run([train_step[step // cgpt if step < 2*cgpt else 2], data_num])
          num_examples += temp
        step += 1
    except tf.errors.OutOfRangeError:
      checkpoint_file = os.path.join(FLAGS.model_dir, FLAGS.model_file)
      saver.save(sess, checkpoint_file)
      print('Train for %d steps, %d examples.' % (step, num_examples))

  sess.close()


def main(_):
  if FLAGS.mode == 0:
    train()
  elif FLAGS.mode == 1:
    eval()
  elif FLAGS.mode == 2:
    merge()
  elif FLAGS.mode == 3:
    stepwise_train()
  elif FLAGS.mode == 4:
    train_double()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--train_file',
    type=int,
    nargs='+',
    default=None,
    help='Train file numbers.'
  )
  parser.add_argument(
    '--eval_file',
    type=int,
    nargs='+',
    default=None,
    help='Test file numbers.'
  )
  parser.add_argument(
    '--model_file',
    type=str,
    default='model_merged.ckpt',
    help='File to save model.'
  )
  parser.add_argument(
    '--init_file',
    type=str,
    default='model_init.ckpt',
    help='File to restore model.'
  )
  parser.add_argument(
    '--mode',
    type=int,
    default=1,
    help='0 for train, 1 for eval, 2 for merge, 3 for stepwise train.'
  )
  parser.add_argument(
    '--lr',
    type=float,
    default=None,
    help='learning rate.'
  )
  parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='Batch size.'
  )
  parser.add_argument(
    '--num_epochs',
    type=int,
    default=2,
    help='Num of epoch.'
  )
  parser.add_argument(
    '--model_list',
    type=str,
    nargs='+',
    default=['model_1.ckpt', 'model_2.ckpt'],
    help='Model files for merge .'
  )
  parser.add_argument(
    '--data_dir',
    type=str,
    default='./input_data/',
    help='Directory for input data'
  )
  parser.add_argument(
    '--model_dir',
    type=str,
    default='./models/',
    help='Directory for storing checkpoint file.'
  )
  parser.add_argument(
    '--summary_dir',
    type=str,
    default='./summary/',
    help='Directory for storing summaries.'
  )
  parser.add_argument(
    '--rd',
    type=int,
    default=1,
    help='Number of current round.'
  )
  parser.add_argument(
    '--gpu',
    type=str,
    default='2',
    help='Number of gpu to use.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  # print(external.FLAGS)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
