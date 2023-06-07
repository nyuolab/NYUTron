Here is a list of steps for finetuning NYUTron for a specific task. For more details see the method section of our paper.

## 1. Data Processing

### Step 1.1: Pull data (code ommitted for license compliance)

Gain access to the hospital's database. This usually requires IRB (Institutional Review Board) approval and support from the hospital's data mamnagement team. 

After getting access, we wrote a sequence of sql scripts to gather clinical notes and labels. To directly pull data into the cluster, we used kerberos authentication and pyodbc connection. The sql scripts are not included in this repo to comply with Langone's datalake license. 

### Step 1.2 - 1.4: Preprocessing 

In this repo, we combined all the preprocessing steps into a class [FinetuneDataPipeline](https://github.com/nyuolab/NYUTron_open/blob/78f0bb5a928ebef878340ca63977993b33c38f2d/src/nyutron/data_utils.py#L629). This pipeline includes the following steps: 

1.2 clean data (```DfSubsetModule```)

1.3 data split (```TrainValTestSplitModule```)

1.4 tokenize text (```PretokenizerModule```)

If your intended usage is the same, you can directly use the pipeline. (See example in [2_make_finetune_datasets.py](../examples/2_make_finetune_datasets.py). For customization, you can edit the config file [toy_example.yaml](../examples/configs/finetune_data_configs/toy_example.yaml). Note that all our examples use synthetic data here, because we cannot publicly release our original dataset without further verification. Our original dataset contains patient health information and requires training and approval for access. If you need access to the original dataset, please contact Shannon Ciprut (shannon.ciprut@nyulangone.org).

If our pipeline does not fit your application, you may create your own data pipeline with ```nyutron``` (by defining a child class of ```PipelineWithHistory``) and potentially reuse some of the data processing module. 

## 2. Training

After getting a preprocessed, tokenized finetune dataset, we pretrained a 100M BERT model from scratch. Our pretraining script is available at [4_finetune.py](../examples/4_finetune.py). For illustration purpose (able to run on cpu), we configured (see [toy_example.yaml](../examples/configs/finetune_data_configs/toy_example.yaml)) the example to finetune a super tiny BERT (2 layers, 2 attention head, 128 hidden size) using a super tiny dataset (a few synthetic examples) for a very short time (1 epoch). Our original config can be found at  [readmission.yaml](examples/configs/finetune_data_configs/readmission.yaml). 

Our pretraining script uses huggingface trainer and is compatible with cpu training and single/multi GPU training. The launch command is:

```python 4_finetune.py```

You can customize configurations by editing [toy_example.yaml](../examples/configs/finetune_data_configs/toy_example.yaml). 

Note that for multi-gpu training, the effective batch size increase by n (where n is the number of GPUs).
