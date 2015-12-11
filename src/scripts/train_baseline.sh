#!/bin/bash -e

TRAINING_DATA=data/tacl2015-training.txt

OUT_DIR=output/baseline/
MODEL=$OUT_DIR/baseline.txt

mkdir -p $OUT_DIR

./src/scripts/train_baseline.py $TRAINING_DATA $MODEL
