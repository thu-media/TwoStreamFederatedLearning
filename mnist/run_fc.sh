#!/bin/bash
cp ./input_data/model_init.ckpt.data-00000-of-00001 ./input_data/model_1m2.ckpt.data-00000-of-00001
cp ./input_data/model_init.ckpt.index ./input_data/model_1m2.ckpt.index
cp ./input_data/model_init.ckpt.meta ./input_data/model_1m2.ckpt.meta

for ((i=1; i<101; i++))
do
    echo "Round $i."
    echo "Train 1."
    python3 mnist.py train_0_4.tfrecords test_0_4.tfrecords model_1.ckpt 0
    echo "Train 2."
    python3 mnist.py train_5_9.tfrecords test_5_9.tfrecords  model_2.ckpt 0
    echo "Merge CKPT."
    python3 merge_checkpoint.py model_1m2.ckpt --model_list model_1.ckpt model_2.ckpt
    echo "Test All."
    python3 mnist.py _ test.tfrecords model_1m2.ckpt 1
    echo "Test 1."
    python3 mnist.py _ test_0_4.tfrecords model_1m2.ckpt 1
    echo "Test 2."
    python3 mnist.py _ test_5_9.tfrecords model_1m2.ckpt 1
done
