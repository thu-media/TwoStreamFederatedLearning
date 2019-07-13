#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys
import subprocess
import json
import random

C = 0.1
ROUND = 100

# Load clients - shards
clients = None
with open('./clients_number.json', 'r', encoding='utf-8') as infile:
    clients = json.load(infile)

distr = [0 for i in range(200)]
with open('./input_data/shards/shards_dis_train.txt', 'r') as infile:
    for line in infile:
        content = line.strip().split()
        distr[int(content[0])] = int(content[-1])


def init():
    subprocess.run(['cp',
                    './input_data/deep_model_init.ckpt.data-00000-of-00001',
                    './input_data/shards/deep_model_merged.ckpt.data-00000-of-00001'])
    subprocess.run(['cp',
                    './input_data/deep_model_init.ckpt.index',
                    './input_data/shards/deep_model_merged.ckpt.index'])
    subprocess.run(['cp',
                    './input_data/deep_model_init.ckpt.meta',
                    './input_data/shards/deep_model_merged.ckpt.meta'])


def train():
    # Select C*100 clients at random
    selected = {}
    shuffle_list = list(range(100))
    random.shuffle(shuffle_list)
    for i in range(int(100 * C)):
        sc = shuffle_list[i]
        selected[i] = clients[str(sc)]

    count = [0 for i in range(10)]
    for idx, shards in selected.items():
        for shard in shards:
            count[distr[shard]] += 1

    # For each client
    process_list = list()
    for idx, shards in selected.items():
        train_files = []
        for shard in shards:
            train_files.append('train_%d.tfrecords' % shard)
        process_list.append(subprocess.Popen([
            'python3', 'mnist_deep.py',
            '--model_file', 'deep_model_%d.ckpt' % idx,
            '--init_file', 'deep_model_merged.ckpt',
            '--test', '0',
            '--batch_size', '10',
            '--train_dir', './input_data/shards',
            '--train_file'] + train_files))
    for p in process_list:
        p.wait()


def merge():
    model_list = []
    for i in range(int(100 * C)):
        model_list.append('deep_model_%d.ckpt' % i)
    subprocess.run(['python3', 'merge_checkpoint.py',
                    'deep_model_merged.ckpt',
                    '--train_dir', './input_data/shards',
                    '--model_list'] + model_list)


def test():
    subprocess.run(['python3', 'mnist_deep.py',
                    '--test_file', 'test.tfrecords',
                    '--model_file', 'deep_model_merged.ckpt',
                    '--test', '1',
                    '--train_dir', './input_data/shards'])


def main():
    init()
    for i in range(1):
        subprocess.run(['echo', 'Round %d.' % i])
        train()
        merge()
        test()


if __name__ == '__main__':
    main()
