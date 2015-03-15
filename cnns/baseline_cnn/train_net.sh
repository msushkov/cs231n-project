#!/usr/bin/env sh

dir=$(pwd)

LOG_FILE=$dir/train_output.txt

cd $CAFFE_ROOT

./build/tools/caffe train -solver $dir/solver.prototxt -gpu 0 > $LOG_FILE

cd $dir
