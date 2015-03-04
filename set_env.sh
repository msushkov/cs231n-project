#!/usr/bin/env sh

# Sets all the necessary environment variables that will be used by other scripts.
# Run this before running anything else.

# data (necessary if we wish to train a network)
# (make sure these directories have the file caffe.txt that lists the image locations and labels)
export TRAIN_DATA_ROOT=/path/to/train/
export VAL_DATA_ROOT=/path/to/val/
export TEST_DATA_ROOT=/path/to/test


# these are necessary only if we are evaluating a trained (or pre-trained) network on some test data
export MODEL_FILE=/root/caffe/models/bvlc_reference_caffenet/deploy.prototxt
export PRETRAINED_NETWORK=/path/to/pretrained/network/caffemodel/file

# mean (set these after running make_mean.sh)
export MEAN_FILE_BINARYPROTO=/path/to/mean/binaryproto
export MEAN_FILE_NPY=/path/to/mean/npy


# these probably won't change
export CAFFE_ROOT=/home/caffe
export LABEL_FILENAME=caffe.txt
export LABEL_KEY=label_key.txt

export TRAIN_LABELS=$TRAIN_DATA_ROOT/$LABEL_FILENAME
export VAL_LABELS=$VAL_DATA_ROOT/$LABEL_FILENAME
export TEST_LABELS=$TEST_DATA_ROOT/$LABEL_FILENAME