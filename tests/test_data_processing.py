import unittest, os
import pandas as pd
from pandas.util.testing import assert_frame_equal
from nyutron.data_utils import (
    DfSubsetModule,
    DataProcessingModuleConfig,
    DataProcessingModuleExchange,
)
from nyutron.data_utils import (
    DataProcessingPipeline,
    TrainValTestSplitModule,
    TrainWordPieceTokenizerModule,
)
from nyutron.data_utils import SplitSentenceModule, PretokenizerModule, GroupTextModule
import logging
from transformers import BertTokenizer
from datasets import Dataset
import nltk


class TestDataProcessing(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_cleanup(self):
        """
        test keeping the correct columns"""
        cleanup = DfSubsetModule(
            DataProcessingModuleConfig(
                kwargs={"keep_cols": ["a"]}, logger=logging.getLogger(__name__)
            )
        )
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        data = DataProcessingModuleExchange(data=df)
        res = cleanup(data)
        assert_frame_equal(res.data, pd.DataFrame({"a": [1, 2, 3]}))
        """
        test keeping the default text column"""
        cleanup = DfSubsetModule(
            DataProcessingModuleConfig(logger=logging.getLogger(__name__))
        )
        df = pd.DataFrame({"a": [1, 2, 3], "text": [4, 5, 6]})
        data = DataProcessingModuleExchange(data=df)
        res = cleanup(data)
        self.assertTrue(res.data.equals(pd.DataFrame({"text": [4, 5, 6]})))
        """
        test saving and loading the data"""
        loaded = pd.read_csv(res.path)
        self.assertTrue(loaded.equals(pd.DataFrame({"text": [4, 5, 6]})))
        """
        test saving and loading the data with a different name"""
        cleanup = DfSubsetModule(
            DataProcessingModuleConfig(
                output_path="./cache/test_dir/test.csv",
                logger=logging.getLogger(__name__),
            )
        )
        res = cleanup(data)
        loaded = pd.read_csv(res.path)
        self.assertTrue(loaded.equals(pd.DataFrame({"text": [4, 5, 6]})))

    def test_debug(self):
        """
        test debug option"""
        df = pd.DataFrame({"a": [1, 2, 3], "text": [4, 5, 6]})
        data = DataProcessingModuleExchange(data=df)
        cleanup = DfSubsetModule(
            DataProcessingModuleConfig(
                debug=True, debug_nrows=1, logger=logging.getLogger(__name__)
            )
        )
        res = cleanup(data)
        self.assertEqual(len(res.data), 1)

    def test_train_val_test_split(self):
        """
        test train val test split"""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "text": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            }
        )
        data = DataProcessingModuleExchange(data=df)
        kwargs = {"train_ratio": 0.5, "val_ratio": 0.3}
        splitter = TrainValTestSplitModule(
            DataProcessingModuleConfig(
                kwargs=kwargs, logger=logging.getLogger(__name__)
            )
        )
        res = splitter(data)
        self.assertEqual(len(res.data["train"]), 5)
        self.assertEqual(len(res.data["val"]), 3)
        self.assertEqual(len(res.data["test"]), 2)

    def test_tokenizer_training(self):
        """test tokenizer module"""
        df = pd.DataFrame({"text": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]})
        data = DataProcessingModuleExchange(data=df)
        kwargs = {
            "vocab_size": 10,
            "min_frequency": 1,
            "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        }
        tokenizer_maker = TrainWordPieceTokenizerModule(
            DataProcessingModuleConfig(
                kwargs=kwargs, logger=logging.getLogger(__name__)
            )
        )
        res = tokenizer_maker(data)
        self.assertTrue(os.path.exists(os.path.join(res.path, "vocab.txt")))

    def test_split_sentence(self):
        """test sentence splitting"""
        # single split dataset
        df = Dataset.from_pandas(
            pd.DataFrame(
                {
                    "text": [
                        "a setence.",
                        "a sentence. another sentence.",
                        "a sentence. another sentence. a third sentence.",
                        "a sentence. another sentence. a third sentence. a fourth sentence.",
                    ]
                }
            )
        )
        print(df["text"])
        data = DataProcessingModuleExchange(data=df)
        kwargs = {"batched": True, "batch_size": 2, "num_proc": 1}
        sentence_splitter = SplitSentenceModule(
            DataProcessingModuleConfig(
                kwargs=kwargs, logger=logging.getLogger(__name__)
            )
        )
        res = sentence_splitter(data)
        answer = [
            "a setence.",
            "a sentence.",
            "another sentence.",
            "a sentence.",
            "another sentence.",
            "a third sentence.",
            "a sentence.",
            "another sentence.",
            "a third sentence.",
            "a fourth sentence.",
        ]
        self.assertEqual(sorted(res.data["sentences"]), sorted(answer))

    def test_pretokenize_and_group_tokens(self):
        df = Dataset.from_pandas(
            pd.DataFrame({"text": ["a setence.", "a sentence. another sentence."]})
        )
        data = DataProcessingModuleExchange(data=df)
        kwargs = {"tokenizer_path": "bert-base-uncased"}
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        pretokenizer = PretokenizerModule(
            DataProcessingModuleConfig(
                kwargs=kwargs, logger=logging.getLogger(__name__)
            )
        )
        res = pretokenizer(data)
        """check that the reconstructed tokenized data is the same as the original"""
        for idx, input_id in enumerate(res.data["input_ids"]):
            reconst = tokenizer.decode(input_id, skip_special_tokens=True)
            self.assertEqual(reconst, df["text"][idx])
        group_tokens = GroupTextModule(
            DataProcessingModuleConfig(
                kwargs=kwargs, logger=logging.getLogger(__name__)
            )
        )
        grouped = group_tokens(res)
        self.assertEqual(len(grouped.data), 1)
        grouped_res = "a setence. a sentence. another sentence."
        self.assertEqual(
            tokenizer.decode(grouped.data["input_ids"][0], skip_special_tokens=True),
            grouped_res,
        )

    def test_pipeline_simple(self):
        """test one module pipeline"""
        cleanup = DfSubsetModule(
            DataProcessingModuleConfig(
                kwargs={"keep_cols": ["a"]}, logger=logging.getLogger(__name__)
            )
        )
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "text": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            }
        )
        data = DataProcessingModuleExchange(data=df)
        pipe = DataProcessingPipeline([cleanup])
        res = pipe(data)
        assert_frame_equal(
            res.data, pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        )
        """test two module pipeline"""
        kwargs = {"train_ratio": 0.5, "val_ratio": 0.3}
        splitter = TrainValTestSplitModule(
            DataProcessingModuleConfig(
                kwargs=kwargs, logger=logging.getLogger(__name__)
            )
        )
        pipe = DataProcessingPipeline([cleanup, splitter])
        res = pipe(data)
        self.assertEqual(len(res.data["train"].columns), 1)
        self.assertEqual(res.data["train"].columns[0], "a")
        self.assertEqual(len(res.data["train"]), 5)
        self.assertEqual(len(res.data["val"]), 3)
        self.assertEqual(len(res.data["test"]), 2)

    def test_pipeline_with_tokenizer(self):
        logger = logging.getLogger(__name__)
        # first module: cleanup
        cleanup = DfSubsetModule(DataProcessingModuleConfig(logger=logger))
        # second module: splitter
        splitter = TrainValTestSplitModule(
            DataProcessingModuleConfig(
                kwargs={"train_ratio": 0.5, "val_ratio": 0.25}, logger=logger
            )
        )
        pipe1 = DataProcessingPipeline([cleanup, splitter])
        # third module: tokenizer
        tok_kwargs = {
            "vocab_size": 3,
            "min_frequency": 1,
            "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        }
        tokenizer_maker = TrainWordPieceTokenizerModule(
            DataProcessingModuleConfig(kwargs=tok_kwargs, logger=logger)
        )
        # fourth module: sentence splitter
        kwargs = {"batched": True, "batch_size": 2, "num_proc": 1}
        sentence_splitter = SplitSentenceModule(
            DataProcessingModuleConfig(kwargs=kwargs, logger=logger)
        )
        pipe3 = DataProcessingPipeline([sentence_splitter])
        # connect four modules with two pipelines
        df = pd.DataFrame(
            {"text": ["a.", "a. b.", "a. b. c.", "a. b. c. d."], "a": [1, 2, 3, 4]}
        )
        data = DataProcessingModuleExchange(data=df)
        res1 = pipe1(data)
        res2 = tokenizer_maker(splitter.extract_split(res1, "train"))
        res3 = pipe3(res1)
        self.assertEqual(len(res3.data["train"].column_names), 1)
        self.assertEqual(res3.data["train"].column_names[0], "sentences")
        self.assertEqual(len(res1.data["train"]), 2)
        self.assertEqual(len(res1.data["val"]), 1)
        self.assertEqual(len(res1.data["test"]), 1)
        # note: for small vocab size (test cases), the trained tokenizer
        # is not guaranteed to have the specified vocab size.
        # In particular, the output vocab size might be larger.
        # reference: https://github.com/huggingface/tokenizers/issues/903
        self.assertTrue(os.path.exists(os.path.join(res2.path, "vocab.txt")))

    def test_entire_pretrain_data_pipeline(self):
        logger = logging.getLogger(__name__)
        # first module: cleanup
        cleanup = DfSubsetModule(DataProcessingModuleConfig(logger=logger))
        # second module: splitter
        splitter = TrainValTestSplitModule(
            DataProcessingModuleConfig(
                kwargs={"train_ratio": 0.5, "val_ratio": 0.25}, logger=logger
            )
        )
        pipe1 = DataProcessingPipeline([cleanup, splitter])
        # third module: tokenizer
        tok_kwargs = {
            "vocab_size": 3,
            "min_frequency": 1,
            "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        }
        tokenizer_maker = TrainWordPieceTokenizerModule(
            DataProcessingModuleConfig(kwargs=tok_kwargs, logger=logger)
        )
        df = pd.DataFrame(
            {"text": ["a.", "a. b.", "a. b. c.", "a. b. c. d."], "a": [1, 2, 3, 4]}
        )
        data = DataProcessingModuleExchange(data=df)
        res1 = pipe1(data)
        res2 = tokenizer_maker(splitter.extract_split(res1, "train"))
        # fourth module: sentence splitter
        kwargs = {"batched": True, "batch_size": 2, "num_proc": 1}
        sentence_splitter = SplitSentenceModule(
            DataProcessingModuleConfig(kwargs=kwargs, logger=logger)
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
                kwargs={"tokenizer_path": res2.path}, logger=logger
            )
        )
        pipe3 = DataProcessingPipeline([sentence_splitter, pretokenizer, grouper])
        # connect four modules with two pipeline
        res3 = pipe3(res1)
        self.assertEqual(
            sorted(res3.data["train"].column_names),
            sorted(
                ["attention_mask", "input_ids", "special_tokens_mask", "token_type_ids"]
            ),
        )
        self.assertEqual(len(res1.data["train"]), 2)
        self.assertEqual(len(res1.data["val"]), 1)
        self.assertEqual(len(res1.data["test"]), 1)
        # after grouper, the 2 training examples are grouped into one example
        self.assertEqual(len(res3.data["train"]), 1)
        # note: for small vocab size (test cases), the trained tokenizer
        # is not guaranteed to have the specified vocab size.
        # In particular, the output vocab size might be larger.
        # reference: https://github.com/huggingface/tokenizers/issues/903
        self.assertTrue(os.path.exists(os.path.join(res2.path, "vocab.txt")))


if __name__ == "__main__":
    nltk.download("punkt")
    unittest.main()
