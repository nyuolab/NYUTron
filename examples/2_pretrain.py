# reference:
# [doc](https://huggingface.co/blog/how-to-train)
# [notebook](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb#scrollTo=HOk4iZ9YZvec)
from transformers import BertForMaskedLM, BertTokenizer
from transformers import BertConfig, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datetime import datetime
from datasets import load_from_disk
import os, torch, wandb
from omegaconf import OmegaConf

entity = "lavender"
project_name = "pretrain_open"
conf = OmegaConf.load("configs/pretrain_configs/toy_example.yaml")

# 1. set environment variable
if torch.cuda.is_available():
    os.environ["CUDA_HOME"] = conf.cuda_home  # export cuda_home
# enable tokenizer parallelization for speedup
os.environ["TOKENIZERS_PARALLELISM"] = str(conf.tokenizer_parallel)
today = datetime.now()
time_id = today.strftime("%d_%m_%Y_%H_%M_%S")
# reference: https://docs.wandb.ai/guides/track/advanced/environment-variables
dir_name = f"data/pretrain_ckpt/nyu_ds_multi_node_{time_id}"  # checkpoint directory
if conf.resume:
    os.environ["WANDB_RESUME"] = "auto"
    dir_name = conf.prev_ckpt

# 2. Initialize LM
# Recreate tokenizer
tokenizer = BertTokenizer.from_pretrained(
    conf.data.tokenizer_dir, max_len=conf.data.max_seq_len
)
# Load Pretokenized dataset
dataset = load_from_disk(conf.data.train_data)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=conf.data.mlm_prob
)
if conf.load_trained or conf.resume:
    print(f"loading trained model at {conf.prev_model}....")
    model = BertForMaskedLM.from_pretrained(conf.prev_model)
    print(f"successfully loaded {conf.prev_model}")
else:
    # Initialize Bert
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,  # conf.data.vocab_size,
        max_position_embeddings=conf.data.max_seq_len,
        num_attention_heads=conf.model.n_attention_heads,
        num_hidden_layers=conf.model.n_hidden_layers,
        hidden_size=conf.model.hidden_size,
        type_vocab_size=conf.data.type_vocab_size,
    )

    # Initialize BERT
    model = BertForMaskedLM(config=config)
    print(f"initialized BERT model with {model.num_parameters()} params")


# 3. Initialize Trainer & start training
if not torch.cuda.is_available():
    conf.training.deepspeed = None
    conf.training.fp16 = False
training_args = TrainingArguments(
    output_dir=dir_name,
    overwrite_output_dir=True,
    num_train_epochs=conf.training.n_epochs,
    per_device_train_batch_size=conf.training.per_device_train_batch_size,
    save_steps=conf.training.save_steps,
    save_total_limit=conf.training.save_total_limit,
    prediction_loss_only=conf.training.prediction_loss_only,
    deepspeed=conf.training.deepspeed,  # added deepspeed supp
    logging_steps=conf.training.logging_steps,
    fp16=conf.training.fp16,
    learning_rate=1e-5,
    warmup_ratio=0.01,
    max_grad_norm=0.5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
)

print("initializing wandb....")
wandb.init(
    project=project_name, entity=entity
)  # reference: https://github.com/wandb/client/issues/1499, https://github.com/huggingface/transformers/pull/10826
wandb.config.update(OmegaConf.to_container(conf))  # update wandb config
print("done init wandb!")

trainer.train(resume_from_checkpoint=conf.resume)
