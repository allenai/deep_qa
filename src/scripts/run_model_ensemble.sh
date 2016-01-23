#!/bin/bash -e

DATA_DIR=data/query/
ENTITY_FILE=$JOINT_DATA_DIR/entities.txt
WORD_FILE=$JOINT_DATA_DIR/words.txt

MODEL_DIR=output/query
BASELINE_MODEL_FILENAME=$MODEL_DIR/baseline.txt
US_JOINT_MODEL_NAME="query_d=300_iter=100_l2=1e-4"
MODEL_FILENAME=$MODEL_DIR/out_$US_JOINT_MODEL_NAME.ser

java -Xmx80000M -cp 'lib/*' com.jayantkrish.jklol.lisp.cli.AmbLisp --args "$MODEL_FILENAME" src/lisp/universal_schema/environment.lisp $BASELINE_MODEL_FILENAME $ENTITY_FILE $WORD_FILE src/lisp/universal_schema/universal_schema.lisp src/lisp/universal_schema/run_ensemble.lisp --interactive --noPrintOptions
