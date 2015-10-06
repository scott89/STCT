#!/usr/bin/env bash

~/caffe-reposity/caffe/.build_release/tools/caffe train \
  -solver lstm_gmm_motion_solver2.prototxt \
  -snapshot lstm_gmm_model2/_iter_30000.solverstate \
  2>&1 | tee log_gmm_2.txt

#~/caffe-reposity/caffe/.build_release/tools/caffe train \
#  -solver lstm_gmm_motion_solver2.prototxt \
#  2>&1 | tee log_gmm_2.txt

