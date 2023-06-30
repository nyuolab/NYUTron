#!/bin/bash
# bash convert_hf_to_onnx.sh MODEL_NAME MODEL_TASK
# Example: $ bash convert_hf_to_onnx.sh bioclinical-bert-finetuned-mtsamples sequence-classification

HF_TRANSFORMERS_IMAGE="huggingface/transformers-pytorch-gpu:latest"
MODEL_DIR="hf_models"
MODEL_NAME=$1
MODEL_TASK=$2

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