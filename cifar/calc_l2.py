# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def read_ckpt(ckpt_file):
    reader = tf.train.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    model = list()
    for key in sorted(var_to_shape_map):
        model.append(reader.get_tensor(key))
    return model

def main():
    ckpt_file = os.path.join('./models/', 'model_1.ckpt')
    model_1 = read_ckpt(ckpt_file)
    ckpt_file = os.path.join('./models/', 'model_merged.ckpt')
    model_2 = read_ckpt(ckpt_file)
    print(model_1[-1][8])
    print(model_2[-1][8])
    loss_1 = tf.reduce_sum([tf.nn.l2_loss(model_1[i] - model_2[i]) for i in range(6, 10)])
    loss_2 = tf.reduce_sum([tf.nn.l2_loss(model_1[i] - model_2[i]) for i in range(10, 16)])
    sess = tf.Session()
    loss_v = sess.run(2 * loss_1 + 20 * loss_2)
    print(loss_v)


if __name__ == '__main__':
    main()
