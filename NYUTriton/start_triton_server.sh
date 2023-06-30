#!/bin/bash
# The -d flag will run in detached mode, but you'll lose the logs
# need to specify the dev flag
# ./start_triton_server.sh triton2202_nemo:latest "device=1" tritonmodelrepo -d

DOCKER_IMAGE=$1
GPU_IDs=$2
MODEL_REPO=`pwd`"/"$3
PROD=$4

#Utility function to cleanup container namespace
function docker_cleanup() {
  docker ps -a|grep ${CONTAINER_NAME}
  dockerNameExist=$?
  if ((${dockerNameExist}==0)) ;then
    docker stop ${CONTAINER_NAME} 
    docker rm ${CONTAINER_NAME}
  fi
}

#Utility function to spin up the docker container
#I'm not a docker wizard... do I need to specify where to publish container ports like this if we're setting --net=host
function docker_run() {
  docker run \
    --name=${CONTAINER_NAME} \
    --gpus ${GPU_IDs} \
    -d \
    -v ${MODEL_REPO}/:/models \
    --ipc=host --net=host \
    -p${HTTPPORT}:${HTTPPORT} -p${GRPCPORT}:${GRPCPORT} -p${METRICSPORT}:${METRICSPORT} \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    ${DOCKER_IMAGE} \
    tritonserver --model-repository=/models \
    --strict-model-config=false \
    --http-port ${HTTPPORT} \
    --grpc-port ${GRPCPORT} \
    --metrics-port ${METRICSPORT}
    # --model-control-mode ${MODELCONTROLMODE}\
    # --repository-poll-secs
    # --log-verbose=1
    # -d \
}

# Main logic
if [ "${PROD}" == "-p" ]; then
  CONTAINER_NAME="NYUTRITON_PROD"
  HTTPPORT=8000
  GRPCPORT=8001
  METRICSPORT=8002
else 
  CONTAINER_NAME="NYUTRITON_DEV"
  HTTPPORT=8005
  GRPCPORT=8006
  METRICSPORT=8007
fi

docker_cleanup
docker_run
echo "STARTED ${CONTAINER_NAME}"