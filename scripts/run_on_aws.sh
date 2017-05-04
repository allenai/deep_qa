#!/bin/bash

ARISTO_BINARY=~/clone/aristo/bin/aristo
ECR_REPOSITORY=896129387501.dkr.ecr.us-west-2.amazonaws.com

CONTAINER_NAME=$1
PARAM_FILE=$2

if [ ! -n "$CONTAINER_NAME" ] || [ ! -n "$PARAM_FILE" ] ; then
  echo "USAGE: ./run_on_aws.sh [CONTAINER_NAME] [PARAM_FILE]"
  exit 1
fi


set -e

# Get temporary ecr login.
eval $(aws --region=us-west-2 ecr get-login)

docker pull $ECR_REPOSITORY/infrastructure/aristo/cuda:8

# Create container - we can't push to a conatiner which doesn't exist,
# unlike bintray, but we also can't create a container which does exist,
# so we have to check if it is there first.
if aws --region=us-west-2 ecr describe-repositories | grep -q " \"repositoryName\": *\"$CONTAINER_NAME\"" ; then
    echo "Repository exists - building image."
else
    echo "Creating repository:"
    aws --region=us-west-2 ecr create-repository --repository-name $CONTAINER_NAME
fi

docker build -t $ECR_REPOSITORY/$CONTAINER_NAME . --build-arg PARAM_FILE=$PARAM_FILE
docker push $ECR_REPOSITORY/$CONTAINER_NAME

$ARISTO_BINARY runonce --gpu $ECR_REPOSITORY/$CONTAINER_NAME
