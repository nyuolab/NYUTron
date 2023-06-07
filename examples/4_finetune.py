import os, wandb, torch
import torch.nn as nn
import torch
from omegaconf import OmegaConf, DictConfig
from transformers import EarlyStoppingCallback, set_seed
from transformers import (
    BertTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_from_disk
from nyutron.train_utils import Subsampler
from datetime import datetime
import hydra
import math, random
import numpy as np
from evaluate import load

entity = "lavender"
softmax = nn.Softmax()


# reference: https://huggingface.co/docs/transformers/main_classes/trainer
def preprocess_logits_for_metrics(logits, labels, binary=True):
    probs = softmax(logits)
    if binary:
        pos_probs = probs[:, 1]
        return pos_probs
    else:
        return probs


def compute_metrics(eval_preds, metric_modules, binary=True):
    res = {}
    preds, labels = eval_preds
    for metric_name, metric in metric_modules.items():
        if metric_name == "roc_auc":
            if binary:
                metric_res = metric.compute(references=labels, prediction_scores=preds)
                res[metric_name] = metric_res[metric_name]
            else:
                ovr_metric_res = metric.compute(
                    references=labels, prediction_scores=preds, multi_class="ovr"
                )
                ovo_metric_res = metric.compute(
                    references=labels, prediction_scores=preds, multi_class="ovo"
                )
                res[f"ovr_{metric_name}"] = ovr_metric_res[metric_name]
                res[f"ovo_{metric_name}"] = ovo_metric_res[metric_name]
    return res


def train(model, data, eval_data, test_data, temporal_test_data, tokenizer, conf):
    binary = conf.data.num_label == 2
    metric_for_best_model = "roc_auc" if binary else "ovo_roc_auc"
    args = TrainingArguments(
        output_dir=conf.logger.save_dir,
        save_strategy=conf.trainer.save_strategy,
        save_steps=conf.trainer.save_steps,
        learning_rate=conf.trainer.lr,
        num_train_epochs=conf.trainer.num_train_epochs,
        weight_decay=conf.trainer.weight_decay,
        logging_strategy=conf.trainer.logging_strategy,
        logging_steps=conf.trainer.logging_steps,
        eval_steps=conf.trainer.eval_steps,
        evaluation_strategy=conf.trainer.evaluation_strategy,
        per_device_train_batch_size=conf.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=conf.trainer.per_device_eval_batch_size,
        load_best_model_at_end=True,
        save_total_limit=conf.trainer.save_total_limit,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        gradient_accumulation_steps=conf.trainer.gradient_accumulation_steps,
        report_to=conf.logger.report_to,
    )

    callbacks = []
    if conf.trainer.early_stop:
        print("using early stopping callbacks")
        early_stopper = EarlyStoppingCallback(early_stopping_patience=5)
        callbacks = callbacks.append(early_stopper)

    metrics = conf.trainer.metric
    metric_modules = {}
    for metric in metrics:
        print(f"loading metric {metric}")
        if metric == "roc_auc" and not binary:
            metric_modules[metric] = load(metric, "multiclass")
        else:
            metric_modules[metric] = load(metric)

    trainer = Trainer(
        model,
        args,
        train_dataset=data,
        eval_dataset=eval_data,
        compute_metrics=lambda x: compute_metrics(x, metric_modules, binary),
        preprocess_logits_for_metrics=lambda logits, label: preprocess_logits_for_metrics(
            logits, label, binary
        ),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # update wandb config
    conf_save_path = os.path.join(conf.logger.save_dir, "config.yaml")
    OmegaConf.save(config=conf, f=conf_save_path)
    print(f"save configs to {conf_save_path}!")

    if conf.logger.report_to == "wandb":
        print("initializing wandb....")
        wandb.init(
            project=conf.logger.project,
            entity=entity,
            name=conf.logger.run_name,
            id=conf.logger.run_id,
        )  # reference: https://github.com/wandb/client/issues/1499, https://github.com/huggingface/transformers/pull/10826
        wandb.config.update(OmegaConf.to_container(conf))  # update wandb config
        print("done init wandb!")

    # start training
    trainer.train()

    # start evaluation (both same-time test and future test)
    res = trainer.evaluate(eval_dataset=test_data)
    print(f"test result: {res}")
    temporal_res = trainer.evaluate(eval_dataset=temporal_test_data)
    print(f"temporal test result: {temporal_res}")
    if conf.logger.report_to == "wandb":
        if binary:
            wandb.log(
                {
                    "test/roc_auc": res["eval_roc_auc"],
                    "test/loss": res["eval_loss"],
                    "temporal_test/roc_auc": temporal_res["eval_roc_auc"],
                    "temporal_test/loss": temporal_res["eval_loss"],
                }
            )
        else:
            wandb.log(
                {
                    "test/ovo_roc_auc": res["eval_ovo_roc_auc"],
                    "test/ovr_roc_auc": res["eval_ovr_roc_auc"],
                    "test/loss": res["eval_loss"],
                    "temporal_test/ovo_roc_auc": temporal_res["eval_ovo_roc_auc"],
                    "temporal_test/ovr_roc_auc": temporal_res["eval_ovr_roc_auc"],
                    "temporal_test/loss": temporal_res["eval_loss"],
                }
            )
    return trainer


# python finetune_launch_binary.py -m run.seed=0,13,24,36,42 data.num_train_samples=100,1000,10000,100000 slurm=a100-all hydra/launcher=submitit_slurm
# @hydra.main(version_base=None, config_path="configs/finetune_configs", config_name="toy_comorbidity")
@hydra.main(
    version_base=None,
    config_path="configs/finetune_configs",
    config_name="toy_readmission",
)
def finetune(conf: DictConfig) -> None:
    # set seed for reproducibility
    torch.manual_seed(conf.run.seed)
    np.random.seed(conf.run.seed)
    random.seed(conf.run.seed)
    set_seed(conf.run.seed)
    print(conf)
    # load tokenizer, model and data
    model = AutoModelForSequenceClassification.from_pretrained(
        conf.model.path, num_labels=conf.data.num_label
    )  # does not set load=True bc we are training a new finetuned model
    tokenizer = BertTokenizer.from_pretrained(conf.data.tokenizer.path)
    print(f"loaded tokenizer from {conf.data.tokenizer.path}")
    full_data = load_from_disk(conf.data.tokenized_data_path)
    print(f"loaded data {full_data} from {conf.data.tokenized_data_path}")
    if conf.data.num_train_samples is not None:
        subsampler = Subsampler(seed=conf.run.seed, data=full_data["train"])
        data = subsampler.subsample(conf.data.num_train_samples)
    else:
        data = full_data["train"]
    if conf.data.num_eval_samples is not None:
        eval_subsampler = Subsampler(seed=conf.run.seed, data=full_data["val"])
        eval_data = eval_subsampler.subsample(conf.data.num_eval_samples)
    else:
        eval_data = full_data["val"]
    if conf.run.debug:
        test_subsampler = Subsampler(seed=conf.run.seed, data=full_data["test"])
        test_data = test_subsampler.subsample(100)
    else:
        test_data = full_data["test"]
    if conf.run.debug:
        test_subsampler = Subsampler(
            seed=conf.run.seed, data=full_data["temporal_test"]
        )
        temporal_test_data = test_subsampler.subsample(100)
    else:
        temporal_test_data = full_data["temporal_test"]

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()  # 8
        # calculate eval interval
        n_examples_per_step = (
            conf.trainer.per_device_train_batch_size
            * n_gpus
            * conf.trainer.gradient_accumulation_steps
        )
        n_steps_per_epoch = math.ceil(len(data) / n_examples_per_step)
        eval_steps = math.ceil(n_steps_per_epoch * conf.trainer.p_eval)
    else:
        # when gpu is not available, assume we are working with tiny toy dataset on cpu and set eval step to 1
        eval_steps = 1
    if eval_steps < 1:
        eval_steps = 1
    conf.trainer.eval_steps = eval_steps
    conf.trainer.save_steps = conf.trainer.eval_steps
    print(f"setting eval_steps and save_steps to {eval_steps}")

    # configure wandb log
    today = datetime.now()
    time_id = today.strftime("%d_%m_%Y_%H_%M_%S")
    conf.logger.run_name = f"{conf.model.pretrained}-{conf.data.num_train_samples}samples-seed{conf.run.seed}_{time_id}"
    if conf.logger.report_to == "wandb":
        conf.logger.run_id = wandb.util.generate_id()
        print(f"wandb run is is {conf.logger.run_id}")
    save_dir = f"{conf.logger.output_dir}/{conf.logger.run_name}"
    conf.logger.save_dir = save_dir
    print(f"result will save to {save_dir}")

    trainer = train(
        model,
        data=data,
        eval_data=eval_data,
        test_data=test_data,
        temporal_test_data=temporal_test_data,
        tokenizer=tokenizer,
        conf=conf,
    )
    return trainer


if __name__ == "__main__":
    finetune()
