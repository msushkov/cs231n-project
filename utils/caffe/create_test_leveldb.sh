#!/usr/bin/env sh
# Create the lmdb inputs
# N.B. set the path to the train + val data dirs in set_env.sh

TOOLS=$CAFFE_ROOT/build/tools

VAL_DATA_ROOT=/root/cs231n-project/test_data/instagram

rm -rf $VAL_DATA_ROOT/val_lmdb

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi


if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT/ \
    $VAL_DATA_ROOT/caffe.txt \
    $VAL_DATA_ROOT/val_lmdb

echo "Done."