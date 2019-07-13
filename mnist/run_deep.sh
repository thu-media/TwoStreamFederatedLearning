#!/bin/bash
cp ./input_data/deep_model_init.ckpt.data-00000-of-00001 ./input_data/deep_model_1m2.ckpt.data-00000-of-00001
cp ./input_data/deep_model_init.ckpt.index ./input_data/deep_model_1m2.ckpt.index
cp ./input_data/deep_model_init.ckpt.meta ./input_data/deep_model_1m2.ckpt.meta

for ((i=1; i<101; i++))
do
    echo "Round $i."
    echo "Train 1."
    python3 mnist_deep.py --train_file train_0_4.tfrecords --test_file test_0_4.tfrecords --model_file deep_model_1.ckpt --init_file deep_model_1m2.ckpt --test 0 --train_dir ./input_data
    echo "Train 2."
    python3 mnist_deep.py --train_file train_5_9.tfrecords --test_file test_5_9.tfrecords --model_file deep_model_2.ckpt --init_file deep_model_1m2.ckpt --test 0 --train_dir ./input_data
    echo "Merge CKPT."
    python3 merge_checkpoint.py deep_model_1m2.ckpt --model_list deep_model_1.ckpt deep_model_2.ckpt
    echo "Test All."
    python3 mnist_deep.py --test_file test.tfrecords --model_file deep_model_1m2.ckpt --train_dir=./input_data
done
