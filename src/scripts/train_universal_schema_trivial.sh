#!/bin/bash -e

DATA_DIR=/home/jayantk/data/universal_schema/clueweb/lf/training_trivial/
MODEL_DIR=/home/jayantk/data/universal_schema/clueweb/models/trivial/

ENTITY_FILE=$DATA_DIR/entities.txt
WORD_FILE=$DATA_DIR/words.txt
LF_FILE=$DATA_DIR/lf.txt

EPOCHS=100
L2_REGULARIZATION=0.0001
L2_REGULARIZATION_FREQ=0.0001
NAME=trivial2
MODEL_OUTPUT=$MODEL_DIR/out_$NAME.ser
LOG_OUTPUT=$MODEL_DIR/log_$NAME.txt

QUERY_JSON=/home/jayantk/data/universal_schema/clueweb/lf/test/annotated_queries_json.txt
RESULTS_DIR=$MODEL_DIR
US_RESULTS=$RESULTS_DIR/results_$NAME.txt

java -cp 'lib/*' com.jayantkrish.jklol.lisp.cli.AmbLisp --optEpochs $EPOCHS --optL2Regularization $L2_REGULARIZATION --optL2RegularizationFrequency $L2_REGULARIZATION_FREQ --args $MODEL_OUTPUT src/lisp/universal_schema/environment.lisp $ENTITY_FILE $WORD_FILE $LF_FILE src/lisp/universal_schema/universal_schema.lisp src/lisp/universal_schema/train_universal_schema.lisp > $LOG_OUTPUT


./src/scripts/us/run_query_test.py $MODEL_OUTPUT $QUERY_JSON $DATA_DIR us > $US_RESULTS