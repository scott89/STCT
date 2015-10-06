#!/usr/bin/env bash

#~/caffe-reposity/caffe/.build_release/tools/caffe train \
#  -solver lstm_gmm_motion_solver.prototxt \
#  -snapshot lstm_gmm_model/_iter_35000.solverstate \
#  2>&1 | tee log_gmm2.txt

~/caffe-reposity/caffe/.build_release/tools/caffe train \
  -solver lstm_gmm_motion_solver60.prototxt \
  2>&1 | tee log_gmm60.txt

