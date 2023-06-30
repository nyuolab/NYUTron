# NYUtriton

author: [Eric K. Oermann](https://github.com/RespectableGlioma), [Anas Abidin](https://anaszain89.github.io/)

NYUtriton (pronounced “nutrition”) is for the Triton deployment of NYUtron. This is an accompanying code repository for paper "Health system scale language models are general purpose clinical prediction engines"

This repository includes code for standing up NYUtron with built in pre-processing for NYUtron as well as interfaces for FHIR and EPIC Nebula.

This ReadMe is focused on running the system within NYU.

**We're currently using OLAB-1 as our production system pending more hardware**

## Requirements
nvidia-docker

## Steps to start system and deploy a model on OLAB-1.

### Copy over codebase and models
Login to OLAB-1 and forward the following ports for the monitoring service:

```bash
ssh USERNAME@IP -L 9090:IP:9090 -L 3000:IP:3000
```

As root user clone the NYUTriton repo and copy over the models you want to deploy into ./hf_models. If the models are from HuggingFace, you'll need git-lfs.

```bash
PAT=<YOUR ACCESS TOKEN>
git clone https://${PAT}@github.com/nyuolab/NYUtriton.git
```

### Convert to accelerators and stage
Convert to accelerators using the conversion script and supplying the model name and the model task (per the HuggingFace model tasks). For example...
```bash
MODEL_NAME=nyutron_readmission
MODEL_TASK=sequence-classification
bash convert_hf_to_trt.sh ${MODEL_NAME} ${MODEL_TASK}
```

Now run the build script to stage the models you want to deploy for deployment
```bash
MODEL_NAME=nyutron_readmission
MODEL_TYPE=onnx
python deploy_model.py --model_name ${MODEL_NAME} --model_type ${MODEL_TYPE}
```

### Deploy and test
Now launch Triton in prod mode with device 1 as the assigned GPU
```bash
./start_triton_server.sh triton2202_nemo:latest "device=1" tritonmodelrepo --p #prod
```

You can build test queries using the serialize_txt.py script and specifying the port, string payload, model name, and model type like below:
```bash
python serialize_txt.py -p 8000 -l "test payload is test payload" -m nyutron_readmission -t onnx
```

or inspect model health and parameters usingthe REST API like so:
```bash
curl -v IP:8000/v2/models/nyutron_readmission_onnx/config
```

### Start monitoring services:
Lastly we can initialize the monitoring services:
```bash
./start_promethus_service.sh --prod
```