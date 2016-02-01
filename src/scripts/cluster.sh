#!/bin/bash
# Spark cluster script.
# For scala2.11, edit:
# https://github.com/amplab/spark-ec2/tree/branch-1.4/scala/init.sh
# https://github.com/amplab/spark-ec2/tree/branch-1.4/scala/init.sh

AWS_PEM_FILE="/home/mattg/.aws/mattg-spark.pem"
NUM_SLAVES=30

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$DIR/..
SCRIPT_NAME=$0

if [ -z "$AWS_PEM_FILE" ]
then
    echo "Variable AWS_PEM_FILE is unset."
    exit 1
fi

SPARK_VERSION=${SPARK_VERSION:-1.4.1}
CLUSTER_NAME=${CLUSTER_NAME:-s2-spark-job-$USER}
REGION=${REGION:-us-west-2}
ZONE=${ZONE:-us-west-2b}
INSTANCE_TYPE=${INSTANCE_TYPE:-m3.xlarge}
VPC=${VPC:-vpc-681db30d}
SUBNET=${SUBNET:-subnet-c49616a1}
NUM_SLAVES=${NUM_SLAVES:-5}

SPARK_TARBALL=spark-$SPARK_VERSION-bin-hadoop2.4.tgz
SPARK_TARBALL_URL=http://apache.cs.utah.edu/spark/spark-$SPARK_VERSION/$SPARK_TARBALL
SPARK_DIR=$ROOT_DIR/tmp/spark/spark-$SPARK_VERSION-bin-hadoop2.4
SPARK_EC2=$SPARK_DIR/ec2/spark-ec2

read -d '' USAGE << EOF
Usage:
  cluster.sh (launch ? | destroy)
  cluster.sh launch --slaves <num_slaves>
  cluster.sh run echo "Hello world"
EOF

function rsync_with_opts {
  rsync -d --delete -z --progress -e "ssh -i $AWS_PEM_FILE" $*
}

function install {
    if [ -d "$SPARK_DIR" ]; then
        echo Spark installed in $SPARK_DIR
    else
        echo Installing Spark in $SPARK_DIR
        echo A one-time procedure.
        mkdir -p $ROOT_DIR/tmp/spark
        (cd $ROOT_DIR/tmp/spark && wget $SPARK_TARBALL_URL && tar -xvzf $SPARK_TARBALL)
    fi
}

function launch {
    install
    $SPARK_EC2 \
        --key-pair=dev-keypair \
        --identity-file=$AWS_PEM_FILE \
        --slaves=$NUM_SLAVES \
        --region=$REGION \
        --zone=$ZONE \
        --instance-type=$INSTANCE_TYPE \
        --vpc-id=$VPC \
        --subnet-id=$SUBNET \
        --private-ips \
        --spark-version=$SPARK_VERSION \
        --copy-aws-credentials \
        --no-ganglia \
        --hadoop-major-version=yarn \
        launch $CLUSTER_NAME
    configure
}

function configure {
  rsync_with_opts src/main/resources/log4j.properties root@$(master):~/spark/conf/
  run mkdir /tmp/spark-events
  # run ephemeral-hdfs/sbin/start-dfs.sh
}

function destroy {
    $SPARK_EC2 \
        --region $REGION \
        --zone $ZONE \
        --delete-groups \
        destroy $CLUSTER_NAME
}

function run {
    echo $*
    ssh -i $AWS_PEM_FILE -t root@$(master) $*
}

function login {
  $SPARK_EC2 --region $REGION --zone $ZONE --private-ips --identity-file=$AWS_PEM_FILE login $CLUSTER_NAME
}

function master {
  local MASTER=$($SPARK_EC2 --region $REGION --zone $ZONE --private-ips get-master $CLUSTER_NAME | tail -n 1)
  echo $MASTER
}

function master_url {
  echo spark://$(master):7077
}

function ui {
    open http://$(master):8080
}

function start {
    JAR=$1
    shift
    CLASS_NAME=$1
    shift
    rsync_with_opts target/scala-2.10/$JAR root@$(master):~/
    run "nohup sh -c \"( (source spark/conf/spark-env.sh && \
        source spark-ec2/ec2-variables.sh && \
        spark/bin/spark-submit --class $CLASS_NAME --master $(master_url) $JAR &> log.txt;
        echo DONE >> log.txt) & )\""
    log
}

function log {
  run less +F log.txt
}

function kill {
    # From ui.
    # APP_ID=$1 broken
    # spark/bin/spark-class org.apache.spark.deploy.Client kill $(master_url) $APP_ID
    echo Please kill jobs using Spark UI.
}

case $1 in
    install)
	    install
	    ;;
    launch)
	    launch
	    ;;
    configure)
        configure
        ;;
    destroy)
	    destroy
	    ;;
    master)
	    master
	    ;;
    ui)
        ui
        ;;
    run)
        shift
        run $*
        ;;
    login)
        login
        ;;
    start)
        shift
        start $*
        ;;
    log)
        shift
        log
        ;;
    kill)
        shift
        kill $*
        ;;
    *)
	    echo "$USAGE"
	    exit 1
esac


