#!/usr/bin/env sh

# Sets all the necessary environment variables that will be used by other scripts.
# Run this before running anything else.

# data
export TRAIN_DATA_ROOT=/path/to/train/
export VAL_DATA_ROOT=/path/to/val/

# caffe
export CAFFE_ROOT=/home/caffe

# these probably won't change
export LABEL_FILENAME=caffe.txt
export LABEL_KEY=label_key.txt
