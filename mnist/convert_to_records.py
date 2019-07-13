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

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import random
import json

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set.images
  labels = data_set.labels
  num_examples = data_set.num_examples

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  ## shards 
  # img_list = []
  # for index in range(num_examples):
  #   img_list.append((int(labels[index]), images[index]))
  # img_list.sort(key=lambda x:x[0])

  # cnt = -1
  # filename = ''
  # writer = None
  # shards_labels = {}
  # shards_dis = open(os.path.join(FLAGS.directory + '/shards', 'shards_dis_%s.txt' % name), 'w')
  # size = 300 if name == 'train' else 50
  # for (label, image) in img_list:
  #   image_raw = image.tostring()
  #   example = tf.train.Example(features=tf.train.Features(feature={
  #       'height': _int64_feature(rows),
  #       'width': _int64_feature(cols),
  #       'depth': _int64_feature(depth),
  #       'label': _int64_feature(label),
  #       'image_raw': _bytes_feature(image_raw)}))
  #   cnt += 1
  #   if cnt % size == 0:
  #     try:
  #       writer.close()
  #     except:
  #       pass
  #     finally:
  #       filename = os.path.join(FLAGS.directory + '/shards',  '%s_%d.tfrecords' % (name, cnt / size))
  #       writer = tf.python_io.TFRecordWriter(filename)
  #   writer.write(example.SerializeToString())

  #   if int(cnt / size) not in shards_labels:
  #     shards_labels[int(cnt / size)] = set()
  #   shards_labels[int(cnt / size)].add(label)
  # for (key, values) in shards_labels.items():
  #   shards_dis.write('%03d shard:' % key)
  #   for v in values:
  #     shards_dis.write(' %d' % v)
  #   shards_dis.write('\n')
  # try:
  #   writer.close()
  #   shards_dis.close()
  # except:
  #   pass


  writer1 = tf.python_io.TFRecordWriter(os.path.join(FLAGS.directory, name + '_even.tfrecords'))
  writer2 = tf.python_io.TFRecordWriter(os.path.join(FLAGS.directory, name + '_odd.tfrecords'))
  # writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.directory, name + '.tfrecords'))
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    label = int(labels[index])
    # writer.write(example.SerializeToString())
    if label % 2 == 0:
        writer1.write(example.SerializeToString())
    else:
        writer2.write(example.SerializeToString())
  writer1.close()
  writer2.close()
  # writer.close()


def random_assign():
  clients = {}
  shuffle_list = list(range(200))
  random.shuffle(shuffle_list)
  for i in range(200)[::2]:
    idx = int(i / 2)
    clients[idx] = []
    clients[idx].append(shuffle_list[i])
    clients[idx].append(shuffle_list[i + 1])
  
  with open(r'./clients_number.json', 'w', encoding='utf-8') as outfile:
    json.dump(clients, outfile, indent=2, separators=(',', ': '))
    

def main(unused_argv):
  # Get the data.
  data_sets = mnist.read_data_sets(FLAGS.directory,
                                   dtype=tf.uint8,
                                   reshape=False,
                                   validation_size=FLAGS.validation_size)
  
  print(data_sets.train.num_examples)  # 60,000
  print(data_sets.test.num_examples)   # 10,000
                              
  # Convert to Examples and write the result to TFRecords.
  convert_to(data_sets.train, 'train')
  convert_to(data_sets.test, 'test')
  # random_assign()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='./input_data',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=0,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
