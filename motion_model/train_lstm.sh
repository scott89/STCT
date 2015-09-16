#!/usr/bin/env bash

#~/caffe-reposity/caffe/.build_release/tools/caffe train \
#  -solver lstm_motion_solver.prototxt \
#  -weights model/_iter_20000.caffemodel \
#  2>&1 | tee log.txt

~/caffe-reposity/caffe/.build_release/tools/caffe train \
  -solver lstm_motion_solver.prototxt \
  2>&1 | tee log.txt

