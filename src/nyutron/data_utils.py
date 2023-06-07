import os, shutil, logging, csv, glob
import pandas as pd
from omegaconf import OmegaConf
from datasets import Dataset, load_from_disk, DatasetDict
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
import time, uuid
from typing import Optional, List, Any, Dict, Union, Iterable
from itertools import chain
from collections import OrderedDict


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class DataProcessingModuleConfig:
    """
    DataProcessingModuleConfig is the configuration class for DataProcessingModule

    Args:

    output_path: the path to save the processed data
    logger: the logger to use
    cache: whether to cache the processed data
    cache_dir: the directory to cache the processed data
    test_load: whether to test loading the processed data
    debug: whether to run in debug mode
    debug_nrows: the number of rows to use in debug mode
    kwargs: other arguments"""

    output_path: Optional[str] = None
    logger: Optional[logging.Logger] = None
    cache: Optional[bool] = False
    cache_dir: Optional[str] = "./cache"
    test_load: Optional[bool] = False
    debug: Optional[bool] = False
    debug_nrows: Optional[int] = 10
    seed: Optional[int] = 42
    kwargs: Optional[Dict] = field(default_factory=dict)


@dataclass
class DataProcessingModuleExchange:
    """
    DataProcessingModuleExchange is the exchange class for DataProcessingModule

    Args:

    path: the path to the data
    data: the data itself
    """

    path: Optional[str] = None
    data: Optional[Any] = None

    def __post_init__(self):
        if self.path is None and self.data is None:
            raise ValueError("path and data cannot be both None")

    def __repr__(self) -> str:
        return f"===============\nDataProcessingModuleExchange: \n path:{self.path} \n data:{self.data}===============\n"


class DataProcessingModule:
    """
    DataProcessingModule is the base class for all data processing modules"""

    def __init__(self, conf: DataProcessingModuleConfig):
        self.conf = conf
        self.logger = self.conf.logger
        self.time_stamp = time.strftime("%Y_%m_%d_%H_%M")
        if self.conf.output_path is None:
            """
            If output_path is not specified, then use the default output path as a combination of cache_dir, processing module name, and time stamp
            """
            id = str(uuid.uuid4())[:8]
            self.conf.output_path = os.path.join(
                self.conf.cache_dir,
                f"after_{self.__class__.__name__}_{self.time_stamp}_{id}",
            )

    def __repr__(self) -> str:
        return f"DataProcessingModule: {self.__class__.__name__}"

    def process(
        self, data: DataProcessingModuleExchange
    ) -> DataProcessingModuleExchange:
        raise NotImplementedError

    def read_pandas_csv(
        self,
        data: DataProcessingModuleExchange,
        rename_cols: Dict[str, str] = None,
        keep_cols: List[str] = None,
    ) -> pd.DataFrame:
        """
        read_pandas_csv reads a pandas dataframe from either a path or a dataframe object
        when debug is True, only read the first debug_nrows rows
        """
        if data.data is not None:
            """
            if data.data is not None, then assume we have a dataframe object"""
            assert type(data.data) == pd.DataFrame
            df = (
                data.data
                if not self.conf.debug
                else data.data.head(self.conf.debug_nrows)
            )
        else:
            """
            if data.data is None, then assume we have a path to a csv file"""
            nrows = self.conf.debug_nrows if self.conf.debug else None
            self.logger.info("reading df....")
            df = pd.read_csv(data.path, index_col=0, nrows=nrows).dropna()
        if rename_cols is not None:
            df = df.rename(columns=rename_cols)
        if keep_cols is not None:
            df = df[keep_cols]
        return df

    def save_pandas_csv(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        save_pandas_csv saves a pandas dataframe to a csv file"""
        if save_path is None:
            save_path = self.conf.output_path
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        df.to_csv(save_path, index=False)
        logger.info(f"done saving df\n{df} to path\n{save_path}!")
        if self.conf.test_load:
            logger.info("testing load....")
            loaded = pd.read_csv(save_path, on_bad_lines="skip")
            logger.info(loaded)

    def prep_save_folder(self, save_path: str):
        """
        make sure the save_path exists, and if the folder is non-empty, clear it"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        target_dir = os.listdir(save_path)
        # If the directory is nonempty, clear it
        if not len(target_dir) == 0:
            files = glob.glob(os.path.join(save_path, "*"))
            for f in files:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                elif os.path.isfile(f):
                    os.remove(f)
                else:
                    raise NotImplementedError(f"unrecognized file type {f}")

    def get_hf_dataset_from_folder(self, data: DataProcessingModuleExchange) -> Dataset:
        """
        reads a dataset from a folder containing csv files for data splits
        """
        if type(data.data) == Dataset:
            return data.data
        else:
            data_dict = {}
            for file_name in glob.glob(os.path.join(data.path, "*.csv")):
                split_name = file_name[:-4].split("/")[-1]
                data_dict[split_name] = file_name
            if len(data_dict) == 0:
                raise ValueError(f"No csv files found in {data.path}")
            print(f"data_dict is {data_dict}")
            return Dataset.from_csv(data_dict)


class DfSubsetModule(DataProcessingModule):
    """
    CleanUpModule is a DataProcessingModule that cleans up a pandas dataframe to keep a subset of columns
    """

    def __init__(self, conf: DataProcessingModuleConfig):
        super().__init__(conf)
        if not self.conf.output_path[:-4] == ".csv":
            self.conf.output_path = self.conf.output_path + ".csv"
        if "keep_cols" in conf.kwargs:
            self.keep_cols = conf.kwargs["keep_cols"]
        else:
            self.keep_cols = ["text"]
            self.logger.info(f"keep_cols not specified, using default {self.keep_cols}")
        if "rename_cols" in conf.kwargs:
            self.rename_cols = conf.kwargs["rename_cols"]
        else:
            self.rename_cols = None

    def __call__(
        self, data: DataProcessingModuleExchange, save_to_disk: bool = True
    ) -> DataProcessingModuleExchange:
        df = self.read_pandas_csv(
            data, rename_cols=self.rename_cols, keep_cols=self.keep_cols
        )
        logger.info(f"readed df {df}")
        if save_to_disk:
            self.save_pandas_csv(df)
            return DataProcessingModuleExchange(self.conf.output_path, df)
        else:
            return DataProcessingModuleExchange(None, df)


class TrainValTestSplitModule(DataProcessingModule):
    """
    SplitModule is a DataProcessingModule that splits a pandas dataframe into train, val, and test sets
    """

    def __init__(self, conf: DataProcessingModuleConfig):
        """
        Extra keyword arguments:
        train_ratio: the ratio of the train set
        val_ratio: the ratio of the val set

        Note: the test set is the remaining data
        """
        super().__init__(conf)
        if "train_ratio" in conf.kwargs and "val_ratio" in conf.kwargs:
            self.train_ratio = conf.kwargs["train_ratio"]
            self.val_ratio = conf.kwargs["val_ratio"]
        else:
            self.train_ratio = 0.8
            self.val_ratio = 0.1
        if "test_ratio" in conf.kwargs:
            assert conf.kwargs["test_ratio"] == 1 - self.train_ratio - self.val_ratio

    def __call__(
        self, data: DataProcessingModuleExchange
    ) -> DataProcessingModuleExchange:
        df = self.read_pandas_csv(data)
        logger.info(f"readed df {df}")
        test_ratio = 1 - self.train_ratio - self.val_ratio
        if test_ratio <= 0:
            raise RuntimeError(f"test_ratio is {test_ratio} <=0")
        train, val_test = train_test_split(
            df, train_size=self.train_ratio, random_state=self.conf.seed
        )
        relative_val_ratio = self.val_ratio / (self.val_ratio + test_ratio)
        if relative_val_ratio <= 0:
            raise RuntimeError(f"relative_val_ratio is {relative_val_ratio} <=0")
        logger.info(f"splitting test and val...")
        val, test = train_test_split(
            val_test, train_size=relative_val_ratio, random_state=self.conf.seed
        )
        split_dict = {"train": train, "val": val, "test": test}
        for split in split_dict:
            logger.info(f"saving split {split}:{split_dict[split]}")
            split_path = os.path.join(self.conf.output_path, split) + ".csv"
            self.save_pandas_csv(split_dict[split], split_path)
            logger.info(f"saved {split_path}!")
        return DataProcessingModuleExchange(self.conf.output_path, split_dict)

    def extract_split(
        self, data: DataProcessingModuleExchange, split: str
    ) -> DataProcessingModuleExchange:
        """
        extracts a split from the data
        """
        return DataProcessingModuleExchange(
            os.path.join(data.path, f"{split}.csv"), data.data[split]
        )


class TrainWordPieceTokenizerModule(DataProcessingModule):
    """TrainWordPieceTokenizerModule is a DataProcessingModule that trains a WordPieceTokenizer on a dataset"""

    def __init__(self, conf: DataProcessingModuleConfig):
        """
        Extra keyword arguments:
        vocab_size: the size of the vocabulary
        min_frequency: the minimum frequency of a token to be included in the vocabulary
        max_seq_len: the maximum sequence length
        padding: whether to pad the sequences
        special_tokens: the special tokens to be added to the vocabulary

        Note: small vocab_size does not directly correspond to the size of the vocabulary
        reference: https://github.com/huggingface/tokenizers/issues/903"""
        super().__init__(conf)
        self.vocab_size = (
            conf.kwargs["vocab_size"] if "vocab_size" in conf.kwargs else 50000
        )
        self.min_frequency = (
            conf.kwargs["min_frequency"] if "min_frequency" in conf.kwargs else 2
        )
        self.max_seq_len = (
            conf.kwargs["max_seq_len"] if "max_seq_len" in conf.kwargs else 512
        )
        self.padding = conf.kwargs["padding"] if "padding" in conf.kwargs else False
        self.special_tokens = (
            conf.kwargs["special_tokens"]
            if "special_tokens" in conf.kwargs
            else ["[SEP]", "[PAD]", "[UNK]", "[MASK]", "[CLS]"]
        )

    def csv2txt(
        self, data: DataProcessingModuleExchange
    ) -> DataProcessingModuleExchange:
        """helper function to convert a csv file to a txt file, which is the format required by the hf tokenizer"""
        df = self.read_pandas_csv(data)
        logger.info(f"loaded {df}")
        txt_path = os.path.join(self.conf.cache_dir, self.time_stamp + ".txt")
        df.to_csv(txt_path, index=False, header=False, quoting=csv.QUOTE_NONE, sep="\n")
        return DataProcessingModuleExchange(txt_path, df)

    def __call__(
        self, data: DataProcessingModuleExchange
    ) -> DataProcessingModuleExchange:
        txt = self.csv2txt(data)
        paths = [txt.path]
        tokenizer = BertWordPieceTokenizer()
        tokenizer.train(
            files=paths,
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
        )
        self.prep_save_folder(self.conf.output_path)
        tokenizer.save_model(self.conf.output_path)
        logger.info(f"tokenizer saved to {self.conf.output_path}")
        return DataProcessingModuleExchange(self.conf.output_path, tokenizer)


class SplitSentenceModule(DataProcessingModule):
    """SplitSentenceModule is a DataProcessingModule that splits sentences, which speeds up pretraining."""

    def __init__(self, conf: DataProcessingModuleConfig):
        """
        Extra keyword arguments:
        splitter: the sentence splitter to be used, currently only nltk is supported
        text_col: the column name of the text column
        batched: whether to batch sentences for splitting, currently only True is supported
        batch_size: the batch size
        num_proc: the number of processes to use for splitting
        """
        super().__init__(conf)
        self.splitter = conf.kwargs["splitter"] if "splitter" in conf.kwargs else "nltk"
        self.text_col = conf.kwargs["text_col"] if "text_col" in conf.kwargs else "text"
        self.batched = conf.kwargs["batched"] if "batched" in conf.kwargs else True
        self.batch_size = (
            conf.kwargs["batch_size"] if "batch_size" in conf.kwargs else 32
        )
        self.num_proc = conf.kwargs["num_proc"] if "num_proc" in conf.kwargs else 32
        if not self.splitter in ["nltk"]:
            raise NotImplementedError(f"splitter {self.splitter} not implemented!")
        if self.batched == False:
            raise NotImplementedError(
                f"current sentence split function only works with batched=True option!"
            )

    def split_sentences(self, examples) -> Dict[str, List[str]]:
        res = []
        example_txt = examples[self.text_col]
        for example in example_txt:
            if self.splitter == "nltk":
                sents = sent_tokenize(example)
            for sent in sents:
                if sent:  # append if not null
                    res.append(sent)
        return {"sentences": res}

    def get_first_split_name(self, dataset_dict: DatasetDict) -> str:
        splits = list(dataset_dict.column_names.keys())
        return splits[0]

    def __call__(
        self, data: DataProcessingModuleExchange
    ) -> DataProcessingModuleExchange:
        dataset = self.get_hf_dataset_from_folder(data)
        logger.info(f"loaded dataset {dataset}")
        split_func = lambda ex: self.split_sentences(ex)
        logger.info("splitting sentences...")
        rm_cols = (
            dataset.column_names
            if type(dataset) == Dataset
            else dataset[self.get_first_split_name(dataset)].column_names
        )
        sentences = dataset.map(
            split_func,
            remove_columns=rm_cols,
            batched=self.batched,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
        )
        self.prep_save_folder(self.conf.output_path)
        sentences.save_to_disk(self.conf.output_path)
        return DataProcessingModuleExchange(self.conf.output_path, sentences)


class PretokenizerModule(DataProcessingModule):
    """
    pretokenize the data using a pretrained tokenizer"""

    def __init__(self, conf: DataProcessingModuleConfig, path_type: str = "hf"):
        """
        Extra keyword arguments:
        tokenizer: the tokenizer to be used
        tokenizer_path: the path to the tokenizer
        sanity_check: whether to do a sanity check on the tokenization result

        Note: either tokenizer or tokenizer_path must be specified"""
        super().__init__(conf)
        if not ("tokenizer" in conf.kwargs or "tokenizer_path" in conf.kwargs):
            raise ValueError("tokenizer not specified")
        elif "tokenizer" in conf.kwargs:
            self.tokenizer = conf.kwargs["tokenizer"]
        elif "tokenizer_path" in conf.kwargs:
            print(f'initializeing tokenizer from path {conf.kwargs["tokenizer_path"]}')
            self.tokenizer = BertTokenizer.from_pretrained(
                conf.kwargs["tokenizer_path"]
            )
        self.text_column_name = (
            conf.kwargs["text_column_name"]
            if "text_column_name" in conf.kwargs
            else "text"
        )
        self.padding = conf.kwargs["padding"] if "padding" in conf.kwargs else False
        self.max_seq_len = (
            conf.kwargs["max_seq_len"] if "max_seq_len" in conf.kwargs else 512
        )
        self.do_sanity_check = (
            True if not "sanity_check" in conf.kwargs else conf.kwargs["sanity_check"]
        )
        self.batched = True if not "batched" in conf.kwargs else conf.kwargs["batched"]
        self.num_proc = 2 if not "num_proc" in conf.kwargs else conf.kwargs["num_proc"]
        self.batch_size = (
            100 if not "batch_size" in conf.kwargs else conf.kwargs["batch_size"]
        )
        if not path_type in ["hf", "csvs"]:
            raise NotImplementedError(f"path_type {path_type} not implemented!")
        self.path_type = path_type

    def tokenize(self, examples, text_column_name, padding, max_seq_length):
        tokenized_inputs = self.tokenizer(
            examples[text_column_name],
            truncation=True,
            is_split_into_words=False,
            padding=padding,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )
        return tokenized_inputs

    def sanity_check(self, tokenized_dataset: Union[Dataset, DatasetDict]):
        """make sure the tokenized dataset is correct"""
        logger.info(f"tokenized dataset is {tokenized_dataset}")
        if type(tokenized_dataset) == DatasetDict:
            splits = list(tokenized_dataset.column_names.keys())
            tokenized_dataset = tokenized_dataset[splits[0]]
        logger.info("=====Examples of Tokenized Data========")
        for i in range(min(5, len(tokenized_dataset))):
            logging.info(self.tokenizer.decode(tokenized_dataset[i]["input_ids"]))
            logging.info("==============")

    def __call__(
        self, data: DataProcessingModuleExchange, save_to_disk: bool = True
    ) -> DataProcessingModuleExchange:
        if type(data.data) == Dataset or type(data.data) == DatasetDict:
            dataset = data.data
        elif type(data.data) == pd.DataFrame:
            dataset = Dataset.from_pandas(data.data)
        elif self.path_type == "hf":
            dataset = load_from_disk(data.path)
        elif self.path_type == "csvs":
            dataset = self.get_hf_dataset_from_folder(data)
        tokenize_func = lambda x: self.tokenize(
            x, self.text_column_name, self.padding, self.max_seq_len
        )
        tokenized_datasets = dataset.map(
            tokenize_func,
            batched=self.batched,
            num_proc=self.num_proc,
            batch_size=self.batch_size,
            remove_columns=[self.text_column_name],
        )
        if self.do_sanity_check:
            self.sanity_check(tokenized_datasets)
        if save_to_disk:
            self.prep_save_folder(self.conf.output_path)
            tokenized_datasets.save_to_disk(self.conf.output_path)
            return DataProcessingModuleExchange(
                self.conf.output_path, tokenized_datasets
            )
        else:
            return DataProcessingModuleExchange(None, tokenized_datasets)


class GroupTextModule(DataProcessingModule):
    """
    group tokenized text into max_seq_len chunks"""

    def __init__(self, conf: DataProcessingModuleConfig):
        super().__init__(conf)
        if not ("tokenizer" in conf.kwargs or "tokenizer_path" in conf.kwargs):
            raise ValueError("tokenizer not specified")
        elif "tokenizer" in conf.kwargs:
            self.tokenizer = conf.kwargs["tokenizer"]
        elif "tokenizer_path" in conf.kwargs:
            self.tokenizer = BertTokenizer.from_pretrained(
                conf.kwargs["tokenizer_path"]
            )
        self.padding = conf.kwargs["padding"] if "padding" in conf.kwargs else False
        self.max_seq_len = (
            conf.kwargs["max_seq_len"] if "max_seq_len" in conf.kwargs else 512
        )
        self.do_sanity_check = (
            True if not "sanity_check" in conf.kwargs else conf.kwargs["sanity_check"]
        )
        self.batched = True if not "batched" in conf.kwargs else conf.kwargs["batched"]
        self.num_proc = 1 if not "num_proc" in conf.kwargs else conf.kwargs["num_proc"]
        self.batch_size = (
            100 if not "batch_size" in conf.kwargs else conf.kwargs["batch_size"]
        )

    # reference: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of max_seq_length.
    def _group_texts(self, examples, max_seq_length):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + max_seq_length]
                for i in range(0, total_length, max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def sanity_check(self, grouped_tokens: Union[Dataset, DatasetDict]):
        """make sure the grouped dataset is correct"""
        logger.info(f"grouped dataset is {grouped_tokens}")
        if type(grouped_tokens) == DatasetDict:
            splits = list(grouped_tokens.column_names.keys())
            grouped_tokens = grouped_tokens[splits[0]]
        logger.info("=====Examples of Grouped Data========")
        for i in range(min(5, len(grouped_tokens))):
            logging.info(self.tokenizer.decode(grouped_tokens[i]["input_ids"]))
            logging.info("==============")

    def __call__(
        self, data: DataProcessingModuleExchange
    ) -> DataProcessingModuleExchange:
        if type(data.data) == Dataset or type(data.data) == DatasetDict:
            dataset = data.data
        else:
            dataset = load_from_disk(data.path)
        group_func = lambda x: self._group_texts(x, self.max_seq_len)
        grouped_tokens = dataset.map(
            group_func,
            batched=self.batched,
            num_proc=self.num_proc,
            load_from_cache_file=self.conf.cache,
            desc=f"Grouping texts in chunks of {self.max_seq_len}",
        )
        grouped_tokens.save_to_disk(self.conf.output_path)
        if self.do_sanity_check:
            self.sanity_check(grouped_tokens)
        return DataProcessingModuleExchange(self.conf.output_path, grouped_tokens)


class DataProcessingPipeline:
    """
    A pipeline that runs a list of data processing modules

    Args:
    processing_funcs: a list of data processing modules"""

    def __init__(self, processing_funcs: List[DataProcessingModule]) -> None:
        self.processing_funcs = processing_funcs
        self.history = []
        self.history_names = []

    # iterate through the processing functions
    def __call__(
        self, input: DataProcessingModuleExchange, return_history: bool = False
    ) -> Iterable[
        Union[DataProcessingModuleExchange, List[DataProcessingModuleExchange]]
    ]:
        cur_data = input
        self.history = [input]
        self.history_names = ["input"]
        for func in self.processing_funcs:
            print(f"*******going through {func}********")
            next_data = func(cur_data)
            self.history.append(next_data)
            self.history_names.append(str(func))
            cur_data = next_data
        if return_history:
            return next_data, self.history
        else:
            return next_data

    def _add_history(
        self,
        additional_names: List[str],
        additioanl_data: List[DataProcessingModuleExchange],
    ):
        self.history += additioanl_data
        self.history_names += additional_names

    def __repr__(self) -> str:
        s = "DataProcessingPipeline:\n"
        s += "****************\n"
        for idx, func in enumerate(self.processing_funcs):
            s += f"step {idx} : {func}\n"
            s += "----------------\n"
        s += "****************"
        return s

    def print_w_offset(self, offset=0) -> str:
        s = "DataProcessingPipeline:\n"
        s += "****************\n"
        for idx, func in enumerate(self.processing_funcs):
            s += f"step {offset+idx} : {func}\n"
            s += "----------------\n"
        s += "****************"
        return s

    def __len__(self):
        return len(self.processing_funcs)


class PipelineWithHistory:
    def __init__(self):
        self.history = OrderedDict()

    def _clear_history(self):
        self.history = OrderedDict()

    def _fill_history(self, pipe):
        for k, v in zip(pipe.history_names, pipe.history):
            self.history[k] = v

    def print_history(self):
        idx = 0
        for k, v in self.history.items():
            print(f"step {idx} {k} output :\n")
            print(f"path: {v.path}")
            print("----------------")
            print(v.data)
            print("================")
            idx += 1

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError(
            "__call__ should be implemented in the child class (pipeline should be callable)"
        )

    def __repr__(self) -> str:
        raise NotImplementedError(
            "__repr__ should be implemented in the child class (pipeline should be printable)"
        )

    def print_output_path(self):
        raise NotImplementedError(
            "print_output_path should be implemented in the child class"
        )


class PretrainDataPipeline(PipelineWithHistory):
    def __init__(
        self,
        conf_path,
        tokenizer_output_path=None,
        tokenized_data_output_path=None,
        split_data_output_path=None,
    ) -> None:
        super().__init__()
        self.conf = OmegaConf.load(conf_path)
        self.tokenizer_output_path = tokenizer_output_path
        self.tokenized_data_output_path = tokenized_data_output_path
        self.split_data_output_path = split_data_output_path
        # 1st module: cleanup
        cleanup = DfSubsetModule(
            DataProcessingModuleConfig(
                logger=logger, kwargs={"keep_cols": [self.conf.clean_up.txt_col_name]}
            )
        )
        # 2nd module: splitter
        self.splitter = TrainValTestSplitModule(
            DataProcessingModuleConfig(
                output_path=split_data_output_path,
                kwargs={
                    "train_ratio": self.conf.data_split.train_ratio,
                    "val_ratio": self.conf.data_split.val_ratio,
                },
                logger=logger,
            )
        )
        self.pipe1 = DataProcessingPipeline([cleanup, self.splitter])
        # 3rd module: tokenizer
        tok_kwargs = {
            "vocab_size": self.conf.train_tokenizer.vocab_size,
            "min_frequency": self.conf.train_tokenizer.min_frequency,
            "special_tokens": OmegaConf.to_container(
                self.conf.train_tokenizer.special_tokens
            ),
        }
        self.tokenizer_maker = TrainWordPieceTokenizerModule(
            DataProcessingModuleConfig(
                output_path=self.tokenizer_output_path, kwargs=tok_kwargs, logger=logger
            )
        )

    def __call__(self, data: DataProcessingModuleExchange):
        self._clear_history()
        res1 = self.pipe1(data)
        self._fill_history(self.pipe1)
        # print(f'agert res1, history is {self.history}')
        res2 = self.tokenizer_maker(self.splitter.extract_split(res1, "train"))
        self.history[str(self.tokenizer_maker)] = res2
        # 4th module: sentence splitter
        sentence_splitter = SplitSentenceModule(
            DataProcessingModuleConfig(
                kwargs={
                    "batched": self.conf.sentencize.batched,
                    "batch_size": self.conf.sentencize.batch_size,
                    "num_proc": self.conf.sentencize.num_proc,
                },
                logger=logger,
            )
        )
        # 5th module: pretokenizer
        pretokenizer = PretokenizerModule(
            DataProcessingModuleConfig(
                kwargs={"tokenizer_path": res2.path, "text_column_name": "sentences"},
                logger=logger,
            )
        )
        # 6th module: group tokens
        grouper = GroupTextModule(
            DataProcessingModuleConfig(
                output_path=self.tokenized_data_output_path,
                kwargs={"tokenizer_path": res2.path},
                logger=logger,
            )
        )
        self.pipe3 = DataProcessingPipeline([sentence_splitter, pretokenizer, grouper])
        # connect four modules with two pipeline
        res3 = self.pipe3(res1)
        self._fill_history(self.pipe3)
        return res3

    def __repr__(self) -> str:
        s = str(self.pipe1) + "\n"
        s += f"step {len(self.pipe1)} : " + str(self.tokenizer_maker) + "\n"
        if hasattr(self, "pipe3"):
            s += self.pipe3.print_w_offset(offset=len(self.pipe1) + 1)
        else:
            s += "second part of the pipe (pretokenize, group tokenized data) is not ready yet, waiting for tokenizer to be trained"
        return s

    def print_output_path(self):
        if self.tokenizer_output_path is not None:
            print(f"tokenizer saved to {self.tokenizer_output_path}")
        if self.tokenized_data_output_path is not None:
            print(f"tokenized data saved to {self.tokenized_data_output_path}")
        if (
            self.tokenizer_output_path is None
            and self.tokenized_data_output_path is None
        ):
            print(f"see print_history() for cached output path.")


class FinetuneDataPipeline(PipelineWithHistory):
    def __init__(
        self, conf_path, split_data_output_path=None, tokenized_data_output_path=None
    ) -> None:
        super().__init__()
        self.conf = OmegaConf.load(conf_path)
        # 1st module: cleanup
        rename_cols = {}
        if not self.conf.clean_up.label_col_name == "label":
            rename_cols[self.conf.clean_up.label_col_name] = "label"
        if not self.conf.clean_up.txt_col_name == "text":
            rename_cols[self.conf.clean_up.txt_col_name] = "text"
        if len(rename_cols) == 0:
            rename_cols = None
        self.cleanup = DfSubsetModule(
            DataProcessingModuleConfig(
                logger=logger,
                kwargs={"keep_cols": ["label", "text"], "rename_cols": rename_cols},
            )
        )
        # 2nd module: splitter
        self.splitter = TrainValTestSplitModule(
            DataProcessingModuleConfig(
                output_path=split_data_output_path,
                kwargs={
                    "train_ratio": self.conf.data_split.train_ratio,
                    "val_ratio": self.conf.data_split.val_ratio,
                },
                logger=logger,
            )
        )
        self.tokenized_data_output_path = tokenized_data_output_path
        self.split_data_output_path = split_data_output_path
        # 3rd module: pretokenizer
        self.pretokenizer = PretokenizerModule(
            DataProcessingModuleConfig(
                output_path=tokenized_data_output_path,
                kwargs={
                    "tokenizer_path": self.conf.pretokenize.tokenizer_path,
                    "text_column_name": "text",
                },
                logger=logger,
            ),
            path_type="csvs",
        )
        self.pipe = DataProcessingPipeline(
            [self.cleanup, self.splitter, self.pretokenizer]
        )

    def add_temporal(self, temporal: DataProcessingModule):
        clean_temporal = self.cleanup(temporal, save_to_disk=False)
        # don't save tokenized temporal to disk, because it would overwrite tokenized train-val-test otherwise
        tokenized_temporal = self.pretokenizer(clean_temporal, save_to_disk=False)
        # log history
        self.pipe._add_history(
            ["clean_temporal", "tokenized_temporal"],
            [clean_temporal, tokenized_temporal],
        )
        return tokenized_temporal

    def __call__(
        self,
        data: DataProcessingModuleExchange,
        temporal_test: DataProcessingModuleExchange = None,
    ):
        self._clear_history()
        res = self.pipe(data)
        if temporal_test is not None:
            temporal_res = self.add_temporal(temporal_test)
            # combine train-val-test with temporal test
            data_dict = {
                "train": res.data["train"],
                "val": res.data["val"],
                "test": res.data["test"],
                "temporal_test": temporal_res.data,
            }
            res = DatasetDict(data_dict)
            res.save_to_disk(self.tokenized_data_output_path)
            self.pipe._add_history(
                ["combined_temporal_test"],
                [
                    DataProcessingModuleExchange(
                        data=res, path=self.tokenized_data_output_path
                    )
                ],
            )
        self._fill_history(self.pipe)
        return DataProcessingModuleExchange(res, self.tokenized_data_output_path)

    def print_output_path(self):
        print(
            f"finetune split saved to {self.split_data_output_path}\ntokenized data saved to {self.tokenized_data_output_path}"
        )

    def __repr__(self) -> str:
        return str(self.pipe)
