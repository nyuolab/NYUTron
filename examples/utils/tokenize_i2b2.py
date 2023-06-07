# Reference:
# * HF MLM [tutorial](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb#scrollTo=q-EIELH43l_T)
# * Stanza Sentence [Split](https://stanfordnlp.github.io/stanza/tokenize.html)
from datasets import load_from_disk, load_dataset
from transformers import BertTokenizerFast
import logging
from omegaconf import OmegaConf


def tokenize_and_align_labels(examples, tokenizer, conf):
    tokenized_inputs = tokenizer(
        examples[conf.text_column_name],
        truncation=conf.truncation,
        is_split_into_words=conf.is_split_into_words,
        padding=conf.padding,
        max_length=conf.max_seq_len,
    )

    labels = []
    for i, label in enumerate(examples[conf.label_column_name]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(conf.special_token_id)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(conf.special_token_id)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def pretokenize(conf):
    ckpt_dir = conf.ckpt_dir
    dataset = load_dataset(conf.load_path)
    logging.info(f"loaded data {dataset}")
    # start tokenization
    tokenizer = BertTokenizerFast.from_pretrained(
        conf.tokenizer_path, max_len=conf.max_seq_len
    )
    logging.info(f"loaded tokenizer {tokenizer}")
    tokenize_func = lambda x: tokenize_and_align_labels(x, tokenizer, conf)
    logging.info("tokneizing data....")
    # tokenization does not support multiprocessing since we used a for loop
    tokenized_datasets = dataset.map(tokenize_func, batched=conf.batched)
    logging.info(f"tokenized dataset is {tokenized_datasets}")
    logging.info("=====Examples of Tokenized Data========")
    for i in range(min(5, len(tokenized_datasets["train"]))):
        logging.info(tokenizer.decode(tokenized_datasets["train"][i]["input_ids"]))
        logging.info("==============")
    # 3. Save to Disk for faster reuse
    tokenized_datasets.save_to_disk(ckpt_dir)
    logging.info(f"saved tokenized dataset to {ckpt_dir}")
    if conf.test_load:
        # test load
        logging.info("testing loading saved ckpt...")
        loaded = load_from_disk(ckpt_dir)
        logging.info(f"dataset is {tokenized_datasets}\nsaved ckpt is {loaded}")
    return ckpt_dir, tokenized_datasets


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    conf_path = "i2b2.yaml"
    config = OmegaConf.load(conf_path)
    pretokenize(config["tokenize"])
