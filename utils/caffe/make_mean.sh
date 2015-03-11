#!/usr/bin/env sh
# Compute the mean image from the training leveldb

# Make sure to run set_env.sh before this.

# these don't need to be changed
TRAIN_LEVELDB=$TRAIN_DATA_ROOT/train_lmdb
MEAN_OUTPUT_FILE=$TRAIN_DATA_ROOT/mean.binaryproto
MEAN_OUTPUT_FILE_NPY=$TRAIN_DATA_ROOT/mean.npy

dir=$(pwd)
cd $CAFFE_ROOT

./build/tools/compute_image_mean $TRAIN_LEVELDB $MEAN_OUTPUT_FILE

cd $dir

python convert_binaryproto_to_npy.py $MEAN_OUTPUT_FILE $MEAN_OUTPUT_FILE_NPY

echo "Done."