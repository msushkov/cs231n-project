#!/usr/bin/env sh

REFERENCE_MODEL=$CAFFE_ROOT/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel

dir=$(pwd)

LOG_FILE=$dir/train_output.txt

cd $CAFFE_ROOT

./build/tools/caffe train -solver $dir/solver.prototxt -weights $REFERENCE_MODEL -gpu 0 2> $LOG_FILE

cd $dir
