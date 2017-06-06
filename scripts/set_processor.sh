#!/bin/bash

USAGE="usage: ./set-processor [gpu|cpu]"

if [ $# != 1 ]; then
    echo "$USAGE"
    exit 1
fi

PROCESSOR=$1
if [ "$PROCESSOR" == "gpu" ]; then
    echo "Setting the processor to '$PROCESSOR'."
    sed -ie 's/^tensorflow-gpu/tensorflow/g' requirements.txt
    sed -ie 's/^tensorflow/tensorflow-gpu/g' requirements.txt
elif [ "$PROCESSOR" == "cpu" ]; then
    echo "Setting the processor to '$PROCESSOR'."
    sed -ie 's/^tensorflow-gpu/tensorflow/g' requirements.txt
else
    echo "Unknown argument: $PROCESSOR"
    echo "$USAGE"
    exit 1
fi
