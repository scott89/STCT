#! /bin/sh 
cd ./video
wget http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Basketball.zip
unzip Basketball -d tmp
mv tmp/Basketball/img Basketball 
rm Basketball.zip tmp -r
