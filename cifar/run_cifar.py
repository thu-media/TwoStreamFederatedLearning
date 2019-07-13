# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import time
import subprocess
import numpy as np


def init():
    subprocess.run([
        'cp',
        './models/model_init.ckpt.data-00000-of-00001',
        './models/model_merged.ckpt.data-00000-of-00001'])
    subprocess.run([
        'cp',
        './models/model_init.ckpt.index',
        './models/model_merged.ckpt.index'])
    subprocess.run([
        'cp',
        './models/model_init.ckpt.meta',
        './models/model_merged.ckpt.meta'])


def train(train_file, model_file, lr, mode, rd, gpu):
    return subprocess.Popen([
        'python3', 'cifar_main.py',
        '--model_file', model_file,
        '--init_file', 'model_merged.ckpt',
        '--mode', mode,
        '--lr', lr,
        '--rd', rd,
        '--gpu', gpu,
        '--train_file'] + train_file)


def eval():
    return subprocess.run([
        'python3', 'cifar_main.py',
        '--model_file', 'model_merged.ckpt',
        '--mode', '1'])


def merge(model_list):
    return subprocess.run([
        'python3', 'cifar_main.py',
        '--model_file', 'model_merged.ckpt',
        '--mode', '2',
        '--lr', '0',
        '--model_list'] + model_list)


def main():
    init()
    lr = 2e-3
    file_nums = [str(i) for i in range(10)]
    for i in range(500):
        subprocess.run(['echo', 'Round %d' % i])
        # str(lr * np.power(0.999, i))
        np.random.shuffle(file_nums)
        p1 = train(train_file=file_nums[:5], model_file='model_1.ckpt',
            lr=str(lr * np.power(0.999, i)), mode='4', rd=str(i + 1), gpu='2')
        # time.sleep(3)
        p2 = train(train_file=file_nums[5:], model_file='model_2.ckpt',
            lr=str(lr * np.power(0.999, i)), mode='4', rd=str(i + 1), gpu='2')
        p1.wait()
        p2.wait()
        merge(['model_1.ckpt', 'model_2.ckpt'])
        eval()


if __name__ == '__main__':
    main()
