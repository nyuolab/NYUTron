import torch
import os
import json
import datasets
from omegaconf import OmegaConf

logger = datasets.logging.get_logger(__name__)

_DATA_DIR = "../../synthetic_data/processed/i2b2"
_TRAINING_FOLDER = "train"
_DEV_FOLDER = "dev"
_TEST_FOLDER = "test"

_DESCRIPTION = """\
I2B2 2012 Clinical Concept Extraction Dataset 
For more details see https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
"""

_CITATION = """
@article{sunEvaluatingTemporalRelations2013,
  title = {Evaluating Temporal Relations in Clinical Text: 2012 I2b2 {{Challenge}}},
  shorttitle = {Evaluating Temporal Relations in Clinical Text},
  author = {Sun, Weiyi and Rumshisky, Anna and Uzuner, Ozlem},
  year = {2013 Sep-Oct},
  journal = {Journal of the American Medical Informatics Association: JAMIA},
  volume = {20},
  number = {5},
  pages = {806--813},
  issn = {1527-974X},
  doi = {10.1136/amiajnl-2013-001628},
  langid = {english},
  pmcid = {PMC3756273},
  pmid = {23564629},
  keywords = {Artificial Intelligence,clinical language processing,Electronic Health Records,Humans,medical language processing,natural language processing,Natural Language Processing,Patient Discharge Summaries,sharedtask challenges,temporal reasoning,Time,Translational Research; Biomedical},
}
"""


class I2B22012Config(datasets.BuilderConfig):
    """BuilderConfig for Conll2003"""

    def __init__(self, **kwargs):
        """BuilderConfig forConll2003.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(I2B22012Config, self).__init__(**kwargs)


class I2B22012(datasets.GeneratorBasedBuilder):
    """."""

    BUILDER_CONFIGS = [
        I2B22012Config(
            name="i2b22012",
            version=datasets.Version("1.0.0"),
            description="I2B2 2012 Clinical Concept Extraction dataset",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "texts": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-OCCURRENCE",
                                "I-OCCURRENCE" "B-CLINICAL_DEPT",
                                "I-CLINICAL_DEPT",
                                "B-EVIDENTIAL",
                                "I-EVIDENTIAL",
                                "B-PROBLEM",
                                "I-PROBLEM",
                                "B-TEST",
                                "I-TEST",
                                "B-TREATMENT",
                                "I-TREATMENT",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        i2b2_path = _DATA_DIR
        data_folders = {
            "train": os.path.join(i2b2_path, _TRAINING_FOLDER),
            "dev": os.path.join(i2b2_path, _DEV_FOLDER),
            "test": os.path.join(i2b2_path, _TEST_FOLDER),
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"folderpath": data_folders["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"folderpath": data_folders["dev"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"folderpath": data_folders["test"]},
            ),
        ]

    def _generate_examples(self, folderpath):
        logger.info("⏳ Generating examples from = %s", folderpath)
        guid = 0
        for file in os.listdir(folderpath):
            with open(os.path.join(folderpath, file), encoding="utf-8") as f:
                texts = []
                ner_tags = []
                for line in f:
                    if line == "" or line == "\n":
                        if texts:
                            yield guid, {
                                "id": str(guid),
                                "texts": texts,
                                "ner_tags": ner_tags,
                            }
                            guid += 1
                            texts = []
                            ner_tags = []
                    else:
                        # conll2003 tokens are space separated
                        splits = line.split(" ")
                        texts.append(splits[0])
                        ner_tags.append(splits[-1].rstrip())
            # last example
            if texts:
                yield guid, {
                    "id": str(guid),
                    "texts": texts,
                    "ner_tags": ner_tags,
                }
