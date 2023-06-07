from transformers import BertTokenizer, EarlyStoppingCallback
from transformers import DataCollatorForTokenClassification
from datasets import load_from_disk
from evaluate import load
import numpy as np
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from omegaconf import OmegaConf
import argparse
import wandb
from datetime import datetime

today = datetime.now()
time_id = today.strftime("%d_%m_%Y_%H_%M_%S")

parser = argparse.ArgumentParser(description="argparse for slurm job")
parser.add_argument(
    "--conf", required=True, metavar="C", type=str, help="path to config file"
)
parser.add_argument(
    "--seed", required=False, type=int, default=42, help="path to config file"
)
args = parser.parse_args()

conf = OmegaConf.load(args.conf)
tokenized_i2b2 = load_from_disk(conf.data.tokenized_data_path)
print(f"loaded dataset {tokenized_i2b2}")
tokenizer = BertTokenizer.from_pretrained(conf.tokenizer.path)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


def model_init():
    return AutoModelForTokenClassification.from_pretrained(
        conf.model.path, num_labels=conf.data.num_label
    )


classlabel = tokenized_i2b2["test"].features[conf.data.label_col_name].feature
metric = load(
    "seqeval", cache_dir=f"./cache/{conf.model.name}_seqeval_results/{time_id}/cache"
)  # reference: https://huggingface.co/spaces/evaluate-metric/seqeval


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    str_labels = []
    str_preds = []
    for label, pred in zip(labels.flatten(), predictions.flatten()):
        if label >= 0:  # only evaluate entity (i.e. labels that are not "O")
            # reference: https://huggingface.co/docs/datasets/v1.10.2/features.html
            str_labels.append(classlabel.int2str(int(label)))
            str_preds.append(classlabel.int2str(int(pred)))
    # keep only 1 value for hyperparam search
    res = metric.compute(predictions=[str_preds], references=[str_labels])
    return res


run_id = wandb.util.generate_id()
wandb.init(
    entity="lavender",
    project="i2b2_toy",
    name=f"{conf.model.name}-seed{args.seed}",
    id=run_id,
)

training_args = TrainingArguments(
    output_dir=f"./ckpts/{conf.model.name}_results/{time_id}",
    evaluation_strategy=conf.trainer.evaluation_strategy,
    learning_rate=conf.trainer.lr,
    per_device_train_batch_size=conf.trainer.per_device_train_batch_size,
    per_device_eval_batch_size=conf.trainer.per_device_eval_batch_size,
    num_train_epochs=conf.trainer.num_train_epochs,
    weight_decay=conf.trainer.weight_decay,
    load_best_model_at_end=True,
    report_to="wandb",  # enable logging to W&B
    metric_for_best_model="overall_f1",
    run_name=f"i2b2-2012-finetuning-{conf.model.name}",  # name of the W&B run (optional)
    seed=args.seed,
)

callbacks = []
if conf.trainer.early_stop:
    print("using early stopping callbacks")
    early_stopper = EarlyStoppingCallback(early_stopping_patience=5)
    callbacks = callbacks.append(early_stopper)

trainer = Trainer(
    model_init=model_init,
    # model=model,
    args=training_args,
    train_dataset=tokenized_i2b2["train"],
    eval_dataset=tokenized_i2b2["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)
trainer.train()
res = trainer.evaluate(eval_dataset=tokenized_i2b2["test"])
print(f"test result: {res}")
wandb.log({"test/eval_overall_f1": res["eval_overall_f1"]})
