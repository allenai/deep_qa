#!/bin/bash -e

DATA_DIR=data/predicate_small/
OUTPUT_DIR=output/
MODEL_DIR=$OUTPUT_DIR/predicate_small

ENTITY_FILE=$DATA_DIR/entities.txt
WORD_FILE=$DATA_DIR/words.txt
LF_FILE=$DATA_DIR/lf.txt
FEATURE_FILE1=data/mid_features_list.txt
FEATURE_FILE2=data/mid_pair_features_list.txt

EPOCHS=100
L2_REGULARIZATION=0.0001
L2_REGULARIZATION_FREQ=0.0001
MODEL_NAME="predicate_d=300_iter=100_l2=1e-4"
MODEL_OUTPUT=$MODEL_DIR/out_$MODEL_NAME.ser
LOG_OUTPUT=$MODEL_DIR/log_$MODEL_NAME.txt

mkdir -p $OUTPUT_DIR
mkdir -p $MODEL_DIR

java -cp 'lib/*' -Xmx160g com.jayantkrish.jklol.lisp.cli.AmbLisp --optEpochs $EPOCHS --optL2Regularization $L2_REGULARIZATION --optL2RegularizationFrequency $L2_REGULARIZATION_FREQ --args $MODEL_OUTPUT src/lisp/environment.lisp $ENTITY_FILE $WORD_FILE $LF_FILE $FEATURE_FILE1 $FEATURE_FILE2 src/lisp/universal_schema.lisp src/lisp/train_predicate_ranking_with_graphs.lisp |tee $LOG_OUTPUT
