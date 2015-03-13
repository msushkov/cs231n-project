#!/usr/bin/env sh

dir=$(pwd)

REFERENCE_MODEL=$dir/reference_model/nin_imagenet.caffemodel

LOG_FILE=$dir/train_output.txt

cd $CAFFE_ROOT

./build/tools/caffe train -solver $dir/solver.prototxt -weights $REFERENCE_MODEL -gpu 0 2> $LOG_FILE

cd $dir
