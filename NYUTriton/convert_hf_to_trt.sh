#!/bin/bash
# bash convert_hf_to_trt.sh MODEL_NAME MODEL_TASK
# Example: $ bash convert_hf_to_trt.sh nyutron_readmission sequence-classification
# This will essentially run the convert_hf_to_onnx script and then immediately convert a TensorRT in the same directory

HF_TRANSFORMERS_IMAGE="huggingface/transformers-pytorch-gpu:latest"
MODEL_DIR="hf_models"
MODEL_NAME=$1
MODEL_TASK=$2
TRT_NGC_IMAGE=nvcr.io/nvidia/tensorrt:22.02-py3
in_file=model.onnx
out_file=model.plan

#Run script
mkdir onnx_models/${MODEL_NAME}

docker run \
    --gpus="device=1" \
    -v `pwd`:/workspace \
    -w=/workspace/ \
    $HF_TRANSFORMERS_IMAGE \
    python3 -m transformers.onnx \
        --model='./'${MODEL_DIR}'/'${MODEL_NAME} \
        --feature="${MODEL_TASK}" \
        ./onnx_models/"${MODEL_NAME}"

cd onnx_models/${MODEL_NAME}

docker run \
  --gpus="device=1" \
  -v `pwd`:/workspace/codes -w=/workspace/codes/ \
  $TRT_NGC_IMAGE \
  trtexec --onnx=$in_file --saveEngine=$out_file

#move into the production directories using teh standard format MODEL_NAME/1/model.ext when ready to use