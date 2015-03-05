#!/usr/bin/env sh

# Sets all the necessary environment variables that will be used by other scripts.
# Run this before running anything else.

# data (necessary if we wish to train a network)
# (make sure these directories have the file caffe.txt that lists the image locations and labels)
export TRAIN_DATA_ROOT=/root/cs231n-project/data/sample/train
export VAL_DATA_ROOT=/root/cs231n-project/data/sample/val
export TEST_DATA_ROOT=/root/cs231n-project/data/sample/test


# these are necessary only if we are evaluating a trained (or pre-trained) network on some test data
export MODEL_FILE=/root/caffe/models/bvlc_reference_caffenet/deploy.prototxt
export PRETRAINED_NETWORK=/root/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel

# mean (set these after running make_mean.sh)
export MEAN_FILE_BINARYPROTO=/root/caffe/data/ilsvrc12/imagenet_mean.binaryproto
export MEAN_FILE_NPY=/root/caffe/data/ilsvrc12/imagenet_mean.npy


# these probably won't change
export CAFFE_ROOT=/home/caffe
export LABEL_FILENAME=caffe.txt
export LABEL_KEY=label_key.txt

export TRAIN_LABELS=$TRAIN_DATA_ROOT/$LABEL_FILENAME
export VAL_LABELS=$VAL_DATA_ROOT/$LABEL_FILENAME
export TEST_LABELS=$TEST_DATA_ROOT/$LABEL_FILENAME