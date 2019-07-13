#! /usr/bin/env python3
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
plt.rcParams['pdf.use14corefonts'] = True
font = {'size': 18, 'family': 'Helvetica'}
plt.rc('font', **font)

color = ('b', 'g', 'r', 'k', 'c', 'm', 'y')
labels = ('MMD Loss', 'No constraint', 'L2 Loss')

def plot_single(args):
    log_file = os.path.join(args.dir, args.infile[0])
    output_file = os.path.join(args.dir, args.outfile)
    data = list()
    with open(log_file, 'r') as infile:
        for line in infile:
            if line.startswith('Num examples'):
                contents = line.strip('.\n').split()
                data.append(float(contents[-1]))
    with open(output_file, 'w') as outfile:
        for item in data:
            outfile.write('%f\n' % item)
    img_file = output_file.replace('.csv', '.png')

    plt.plot([i + 1 for (i, v) in enumerate(data)], data, 'k-')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.savefig(img_file, format='png', dpi=600)

def plot_multiple(args):
    log_file = [os.path.join(args.dir, f) for f in args.infile]
    output_file = os.path.join(args.dir, args.outfile)
    all_data = list()
    for filename in log_file:
        with open(filename, 'r') as infile:
            data = list()
            for line in infile:
                if line.startswith('Num examples'):
                    contents = line.strip('.\n').split()
                    data.append(float(contents[-1]))
            all_data.append(data)
    all_data = np.asarray(all_data)
    for idx, data in enumerate(all_data):
        plt.plot([i + 1 for i in range(50)], data[:50], '{}-'.format(color[idx]),
                 linewidth=1, markersize=1, label=labels[idx])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=18)
    plt.xlabel('Communication Rounds', fontsize=18)
    plt.ylabel('Test Accuracy', fontsize=18)
    plt.savefig(output_file, dpi=600)

def main(args):
    if args.mode == 0:
        plot_single(args)
    elif args.mode == 1:
        plot_multiple(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'infile',
        type=str,
        nargs='+',
        help='Log filename for analyzing.'
    )
    parser.add_argument(
        '-m',
        '--mode',
        type=int,
        default=0,
        help='0 for single .txt file, 1 for multiple .csv file'
    )
    parser.add_argument(
        '-o',
        '--outfile',
        type=str,
        help='Output filename.'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default='./log',
        help='dir for log file and output file.'
    )

    args = parser.parse_args()
    if len(args.infile) == 1 and not args.outfile:
        args.outfile = args.infile[0].replace('.txt', '.csv')
    main(args)
