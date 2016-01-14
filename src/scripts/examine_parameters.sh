#!/bin/bash -e

DATA_DIR=data/predicate_small/
OUTPUT_DIR=output/
MODEL_DIR=$OUTPUT_DIR/predicate_small

ENTITY_FILE=$DATA_DIR/entities.txt
WORD_FILE=$DATA_DIR/words.txt

BASELINE_MODEL_FILENAME=$OUTPUT_DIR/baseline_small/baseline.txt

EXAMINE_PARAMS_FILE=src/lisp/examine_parameters.lisp
USCHEMA_FILE=src/lisp/universal_schema_with_graphs.lisp
#RUN_FILE=src/lisp/run_universal_schema.lisp
#RUN_FILE="$BASELINE_MODEL_FILENAME src/lisp/run_ensemble_with_graphs.lisp"

MODEL_NAME="predicate_d=300_iter=100_l2=1e-4"
MODEL_OUTPUT=$MODEL_DIR/out_$MODEL_NAME.ser

mkdir -p $OUTPUT_DIR
mkdir -p $MODEL_DIR

java -cp 'lib/*' -Xmx160g com.jayantkrish.jklol.lisp.cli.AmbLisp --args $MODEL_OUTPUT src/lisp/environment.lisp $ENTITY_FILE $WORD_FILE $USCHEMA_FILE $RUN_FILE $EXAMINE_PARAMS_FILE
