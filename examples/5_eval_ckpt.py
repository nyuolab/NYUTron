"""
evaluate ckpt model
"""
from transformers import (
    BertTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
from evaluate import load
from omegaconf import DictConfig
from torch import nn
import numpy as np
import wandb
import hydra


@hydra.main(
    version_base=None,
    config_path="configs/finetune_configs",
    config_name="toy_readmission",
)
def eval_ckpt(conf: DictConfig) -> None:
    data_split = "val"
    data_path = "data/finetune/toy_readmission/tokenized"
    ckpt_path = "data/pretrain_ckpt/toy_example/checkpoint-1"

    tokenizer = BertTokenizer.from_pretrained(conf.data.tokenizer.path)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_path, num_labels=conf.data.num_label
    )
    data = load_from_disk(data_path)

    softmax = nn.Softmax()

    # reference: https://huggingface.co/docs/transformers/main_classes/trainer
    def preprocess_logits_for_metrics(logits, labels):
        probs = softmax(logits)
        # pos_probs = probs[:,1]
        return probs

    precision = load("precision")
    recall = load("recall")

    def compute_metrics(eval_preds):
        res = {}
        probs, labels = eval_preds
        np.save("nyutron_probs.npy", probs)
        np.save("nyutron_labels.npy", labels)
        # keep only 1 value for hyperparam search
        wandb.log(
            {
                "prognostic_roc": wandb.plot.roc_curve(
                    labels, probs, labels=[0, 1], classes_to_plot=[1]
                )
            }
        )  # visualize auc curve in wandb
        wandb.log(
            {
                "prognostic_pr": wandb.plot.pr_curve(
                    labels, probs, labels=[0, 1], classes_to_plot=[1]
                )
            }
        )
        for threshold in np.arange(0.01, 1, 0.01):
            preds = (probs[:, 1] >= threshold).astype(int)
            precision_res = precision.compute(
                references=labels, predictions=preds, pos_label=1
            )
            res[f"precision_at_{threshold}"] = precision_res["precision"]
            recall_res = recall.compute(
                references=labels, predictions=preds, pos_label=1
            )
            res[f"recall_at_{threshold}"] = recall_res["recall"]
        return res

    train_args = TrainingArguments(
        output_dir="eval_ckpt_log", per_device_eval_batch_size=32
    )

    trainer = Trainer(
        model,
        train_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        tokenizer=tokenizer,
    )

    wandb.init(project="eval_threshold", name="test_eval")
    if data_split is None:
        trainer.evaluate(eval_dataset=data)
    else:
        trainer.evaluate(eval_dataset=data[data_split])


if __name__ == "__main__":
    eval_ckpt()
