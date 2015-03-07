#!/usr/bin/env sh

REFERENCE_MODEL=$CAFFE_ROOT/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel

dir=$(pwd)

cd $CAFFE_ROOT

./build/tools/caffe train -solver solver.prototxt -weights $REFERENCE_MODEL -gpu 0

cd dir
