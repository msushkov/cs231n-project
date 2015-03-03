#!/usr/bin/env sh
# Compute the mean image from the training leveldb

TOOLS=/home/caffe/build/tools
TRAIN_LEVELDB=/path/to/train/leveldb
MEAN_OUTPUT_FILE=/path/to/mean/binaryproto

.$TOOLS/compute_image_mean $TRAIN_LEVELDB $MEAN_OUTPUT_FILE

echo "Done."