#!/bin/bash -e

DATA_DIR=data/query_small/
ENTITY_FILE=$DATA_DIR/entities.txt
WORD_FILE=$DATA_DIR/words.txt

MODEL_DIR=output/query_small
US_JOINT_MODEL_NAME="query_d=300_iter=100_l2=1e-4"
MODEL_FILENAME=$MODEL_DIR/out_$US_JOINT_MODEL_NAME.ser

java -Xmx10000M -cp 'lib/*' com.jayantkrish.jklol.lisp.cli.AmbLisp --args "$MODEL_FILENAME" src/lisp/environment.lisp $ENTITY_FILE $WORD_FILE src/lisp/universal_schema.lisp src/lisp/run_universal_schema.lisp --interactive --noPrintOptions
