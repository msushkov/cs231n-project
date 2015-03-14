#!/usr/bin/env sh
# Create the lmdb inputs
# N.B. set the path to the train + val data dirs in set_env.sh

TOOLS=$CAFFE_ROOT/build/tools

# delete val_lmdb and train_lmdb to avoid errors
rm -rf $TRAIN_DATA_ROOT1/train_lmdb

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

if [ ! -d "$TRAIN_DATA_ROOT1" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT1/ \
    $TRAIN_DATA_ROOT1/caffe.txt \
    $TRAIN_DATA_ROOT1/train_lmdb

echo "Done."