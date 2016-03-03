#!/bin/bash -e

TRAINING_DATA=data/tacl2015-training-sample.txt

OUT_DIR=output/baseline_small/
MODEL=$OUT_DIR/baseline.txt

mkdir -p $OUT_DIR

./src/scripts/train_baseline.py $TRAINING_DATA $MODEL
