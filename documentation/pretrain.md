Here is a list of steps for pretraining NYUTron. For more details see the method section of our paper.

## 1. Data Processing

### Step 1.1: Pull data (code ommitted for license compliance)

Gain access to the hospital's database. This usually requires IRB (Institutional Review Board) approval and support from the hospital's data mamnagement team. 

After getting access, we wrote a sequence of sql scripts to gather clinical notes based on time and notes authors. To directly pull data into the cluster, we used kerberos authentication and pyodbc connection. The sql scripts are not included in this repo to comply with Langone's datalake license. 

### Step 1.2 - 1.7: Preprocessing 

In this repo, we combined all the preprocessing steps into a class [PretrainDataPipeline](https://github.com/nyuolab/NYUTron_open/blob/78f0bb5a928ebef878340ca63977993b33c38f2d/src/nyutron/data_utils.py#L568). This pipeline includes the following steps: 

1.2 clean data (```DfSubsetModule```)

1.3 data split (```TrainValTestSplitModule```)

1.4 train tokenizer with data from train split (```TrainWordPieceTokenizerModule```)

1.5 split text into sentences (```SplitSentenceModule```)

1.6 tokenize text (```PretokenizerModule```)

1.7: group tokenized text to save space (```GroupTextModule```)

If your intended usage is the same, you can directly use the pipeline. (See example in [1_make_pretrain_dataset.py](../examples/1_make_pretrain_dataset.py). For customization, you can edit the config file [toy_example.yaml](../examples/configs/pretrain_data_configs/toy_example.yaml)). Note that all our examples use synthetic data here, because we cannot publicly release our original dataset without further verification. Our original dataset contains patient health information and requires training and approval for access. If you need access to the original dataset, please contact Shannon Ciprut (shannon.ciprut@nyulangone.org).

If our pipeline does not fit your application, you may create your own data pipeline with ```nyutron``` (by defining a child class of ```PipelineWithHistory``) and potentially reuse some of the data processing module. 

## 2. Training

After getting a preprocessed, tokenized pretrain dataset, we pretrained a 100M BERT model from scratch. Our pretraining script is available at [2_pretrain.py](../examples/2_pretrain.py). For illustration purpose (able to run on cpu), we configured (see [toy_example.yaml](../examples/configs/pretrain_data_configs/toy_example.yaml)) the example to train a super tiny BERT (2 layers, 2 attention head, 128 hidden size) using a super tiny dataset (a few synthetic examples) for an unrealistically short time (1 epoch). Our original config can be found at  [nyutron_all.yaml](../examples/configs/pretrain_configs/nyutron_all.yaml). 

Our pretraining script uses huggingface trainer and is compatible with cpu training, single/multi GPU training, and multinode training wity deepspeed acceleration.  

### CPU Training and Single/Multi-GPU Training

```python 2_pretrain.py```

You can customize configurations by editing [toy_example.yaml](../examples/configs/pretrain_configs/toy_example.yaml). 

Note that for multi-gpu training, the effective batch size increase by n (where n is the number of GPUs).

### Multi-Node Training with Deepspeed acceleration

```sbatch 2_launch_pretrain_multinode.sh```

Apart from the yaml file, you can customize deepspeed setup by editing [deepspeed_config_multinode.json](../examples/configs/pretrain_configs/deepspeed_config_multinode.json) and [hostfile](../examples/configs/pretrain_configs/hostfile).  

Note that deepspeed installation might require building from source for your cluster's specific cuda architecture. If normal install does not work, you can check cuda architecture with 

1. ``` CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())" ```
2. clone deepspeed's github repo, ```cd``` to that repo
3. install from source with
``` TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \ --global-option="build_ext" --global-option="-j8" --no-cache -v \ --disable-pip-version-check 2>&1 | tee build.log ```. 

## Extra Tips for working with hpc

### Loading modules 

Modules are system-level software that cannot be installed with conda. For our pretrain script to work on Langone's cluster, we need to load the following modules ```cuda/11.4```, ```gcc/10.2.0```, ```nccl```. These modules may be named differently or not installed on your cluster. In that case you need to contact hpc admins for help. 

### Establishing database connection
Directly downloading data into cluster requires connection to the database system, which is prohibited by some hpc systems for security purpose. For Langone's hpc, there is a specific type of cpu nodes (datamover) that allows this connection with pyodbc support. If you have similar issues, you need to contact hpc admins or database admins for help.

### Using jupyter notebooks

Jupyter notebooks cannot be opened directly when using hpc. Some systems have GUI support service (such as nyu's Open OnDemand) that enable user to connect to hpc and launch notebooks from a webpage. 

I personally use port forwarding. This method sends contents hosted on a hpc node's port to a port in the user's local machine. To use this method, you need to do 3 things:

1. on hpc: launch notebook to a specific port ```jupter notebook --no-browser --port=HPC_PORT_ID --ip=NODE_NAME```
2. on local machine, set up port fowarding from HPC to local ```ssh -N -L USERNAME@HPC.ORG LOCAL_PORT_ID:NODE_NAME:HPC_PORT_ID```
3. on local machine, open browser at localhost:LOCAL_PORT_ID. It should be a notebook page asking for secruity token. This token can be found on your terminal connected to hpc. 

