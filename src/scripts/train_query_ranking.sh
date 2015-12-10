#!/bin/bash -e

DATA_DIR=data/query_small/
OUTPUT_DIR=output/
MODEL_DIR=$OUTPUT_DIR/query_small/

ENTITY_FILE=$DATA_DIR/entities.txt
WORD_FILE=$DATA_DIR/words.txt
LF_FILE=$DATA_DIR/lf.txt

EPOCHS=100
L2_REGULARIZATION=0.0001
L2_REGULARIZATION_FREQ=0.0001
MODEL_NAME="query_d=300_iter=100_l2=1e-4"
MODEL_OUTPUT=$MODEL_DIR/out_$MODEL_NAME.ser
LOG_OUTPUT=$MODEL_DIR/log_$MODEL_NAME.txt

mkdir -p $OUTPUT_DIR
mkdir -p $MODEL_DIR

java -cp 'lib/*' -Xmx10000M com.jayantkrish.jklol.lisp.cli.AmbLisp --optEpochs $EPOCHS --optL2Regularization $L2_REGULARIZATION --optL2RegularizationFrequency $L2_REGULARIZATION_FREQ --args $MODEL_OUTPUT src/lisp/environment.lisp $ENTITY_FILE $WORD_FILE $LF_FILE src/lisp/universal_schema.lisp src/lisp/train_query_ranking.lisp > $LOG_OUTPUT
