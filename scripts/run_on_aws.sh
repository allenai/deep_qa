#!/bin/bash

ARISTO_BINARY=~/clone/aristo/bin/aristo

CONTAINER_NAME=$1
PARAM_FILE=$2

if [ ! -n "$CONTAINER_NAME" ] || [ ! -n "$PARAM_FILE" ] ; then
  echo "USAGE: ./run_on_aws.sh [CONTAINER_NAME] [PARAM_FILE]"
  exit 1
fi


set -e

docker pull allenai-docker-private-docker.bintray.io/cuda:8

docker build -t allenai-docker-private-docker.bintray.io/$CONTAINER_NAME . --build-arg PARAM_FILE=$PARAM_FILE

docker push allenai-docker-private-docker.bintray.io/$CONTAINER_NAME

$ARISTO_BINARY runonce --gpu allenai-docker-private-docker.bintray.io/$CONTAINER_NAME
