# TRAIN_DIR="--train_dir=./input_data"
# MODEL_LIST=("deep_model_1" "deep_model_2")
# TEST_LIST=("test" "test_even" "test_odd" "test_0_4" "test_5_9" "test_0_6" "test_7_9" "test_3_9")
# for MODEL_FILE in ${MODEL_LIST[@]}
# do
#     for TEST_FILE in ${TEST_LIST[@]}
#     do
#         python3 mnist_deep.py --model_file $MODEL_FILE.ckpt --test_file $TEST_FILE.tfrecords $TRAIN_DIR
#     done
# done


MODEL_LIST=("model_1" "model_2")
TEST_LIST=("test" "test_0_4" "test_5_9" "test_0_6" "test_7_9" "test_3_9")
for MODEL_FILE in ${MODEL_LIST[@]}
do
    for TEST_FILE in ${TEST_LIST[@]}
    do
        python3 mnist.py train.tfrecords $TEST_FILE.tfrecords $MODEL_FILE.ckpt 1
    done
done