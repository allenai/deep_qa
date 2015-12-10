#!/bin/bash -e

TRAINING_DATA=data/tacl2015-training.txt
WORD_COUNT_THRESHOLD=5

P_OUT_DIR=data/predicate
P_ENTITIES=$P_OUT_DIR/entities.txt
P_WORDS=$P_OUT_DIR/words.txt
P_LF=$P_OUT_DIR/lf.txt

echo "Generating predicate ranking data..."
mkdir -p $P_OUT_DIR
./src/scripts/generate_entity_tuples.py $TRAINING_DATA $P_ENTITIES $P_WORDS $P_LF $WORD_COUNT_THRESHOLD 0 0 0

Q_OUT_DIR=data/query
Q_ENTITIES=$Q_OUT_DIR/entities.txt
Q_WORDS=$Q_OUT_DIR/words.txt
Q_LF=$Q_OUT_DIR/lf.txt

echo "Generating query ranking data..."
mkdir -p $Q_OUT_DIR
./src/scripts/generate_entity_tuples.py $TRAINING_DATA $Q_ENTITIES $Q_WORDS $Q_LF $WORD_COUNT_THRESHOLD 0 0 1

## Generate additional, smaller training files for messing around by subsampling the training data
TRAINING_DATA_SAMPLE=data/tacl2015-training-sample.txt
head -n100000 $TRAINING_DATA > $TRAINING_DATA_SAMPLE
WORD_COUNT_THRESHOLD=2

P_OUT_DIR=data/predicate_small
P_ENTITIES=$P_OUT_DIR/entities.txt
P_WORDS=$P_OUT_DIR/words.txt
P_LF=$P_OUT_DIR/lf.txt

echo "Generating predicate ranking data (small) ..."
mkdir -p $P_OUT_DIR
./src/scripts/generate_entity_tuples.py $TRAINING_DATA_SAMPLE $P_ENTITIES $P_WORDS $P_LF $WORD_COUNT_THRESHOLD 0 0 0

Q_OUT_DIR=data/query_small
Q_ENTITIES=$Q_OUT_DIR/entities.txt
Q_WORDS=$Q_OUT_DIR/words.txt
Q_LF=$Q_OUT_DIR/lf.txt

echo "Generating query ranking data (small) ..."
mkdir -p $Q_OUT_DIR
./src/scripts/generate_entity_tuples.py $TRAINING_DATA_SAMPLE $Q_ENTITIES $Q_WORDS $Q_LF $WORD_COUNT_THRESHOLD 0 0 1
