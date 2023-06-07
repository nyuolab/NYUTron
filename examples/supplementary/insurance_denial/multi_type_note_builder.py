import csv
import json
import os
import datasets
import pandas as pd
import numpy as np

_DESCRIPTION = """\
Dataset for predicting different types of insurance denial from discharge notes / history and physical notes
"""

_LOCAL_PATHS = {
    "tiny_denials": "/gpfs/data/oermannlab/users/lavender/use-case-nyu-claims-denials/tiny_notes",
    "claims_denials": "/gpfs/data/oermannlab/users/lavender/use-case-nyu-claims-denials/notes",
}

_LABEL_PATH = "/gpfs/data/oermannlab/users/lavender/use-case-nyu-claims-denials/All_Campus_Denial_Data_for_Model_Summary_Final_V2.xlsx"
_HOMEPAGE = ""
_LICENSE = ""
_CITATION = ""

_LABEL_COL = "readmitted_in_30_days"


def id_map(x):
    return x


# task 1 and task 2 share the same label: 1 for initial or final denial, 0 otherwise
# task 1 uses d/c notes for prediction and task 2 uses H&P notes for prediction
def task1_map(x):
    if x == "FINAL-ADVERSE DETERMINATION" or x == "FINAL-FAVORABLE DETERMINATION":
        return 1
    elif x == "No Denial":
        return 0
    else:
        return -1  # invalid


# task 3 and task 4 share the same label: 1 for final denial, 0 otherwise
# task 3 uses d/c notes for prediction and task 4 uses H&P notes for prediction
def task3_map(x):
    if x == "FINAL-ADVERSE DETERMINATION":
        return 1
    elif x == "No Denial" or x == "FINAL-FAVORABLE DETERMINATION":
        return 0
    else:
        return -1  # invalid


class MultiNoteConfig(datasets.BuilderConfig):
    def __init__(
        self,
        notes_path,
        label_path,
        note_type=["Discharge Narrative"],
        label_map=id_map,
        trunc_len=400,
        nrows=-1,
        **kwargs,
    ):
        """BuilderConfig for InContext Readmission."""
        super(MultiNoteConfig, self).__init__(**kwargs)
        self.note_type = note_type
        self.trunc_len = trunc_len
        self.label_map = label_map
        self.notes_path = notes_path
        self.label_path = label_path
        self.nrows = nrows


class MultiNoteReadmissionPrediction(datasets.GeneratorBasedBuilder):
    # example of builder with multiple configs https://huggingface.co/datasets/super_glue/blob/main/super_glue.py
    """Predicting readmission from some context"""
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        MultiNoteConfig(
            name="debug",
            version=VERSION,
            description="small dataset for debugging",
            note_type=["Discharge Narrative"],
            label_map=task1_map,
            trunc_len=400,
            notes_path=_LOCAL_PATHS["tiny_denials"],
            label_path=_LABEL_PATH,
            nrows=-1,
        ),
        MultiNoteConfig(
            name="task1_test",
            version=VERSION,
            description="debug dataset for task 1",
            note_type=["Discharge Narrative"],
            label_map=task1_map,
            trunc_len=400,
            notes_path=_LOCAL_PATHS["claims_denials"],
            label_path=_LABEL_PATH,
            nrows=100,
        ),
        MultiNoteConfig(
            name="task1",
            version=VERSION,
            description="full dataset for task 1",
            note_type=["Discharge Narrative"],
            label_map=task1_map,
            trunc_len=400,
            notes_path=_LOCAL_PATHS["claims_denials"],
            label_path=_LABEL_PATH,
            nrows=-1,
        ),
        MultiNoteConfig(
            name="task2_test",
            version=VERSION,
            description="debug dataset for task 2",
            note_type=["H&P"],
            label_map=task1_map,
            trunc_len=400,
            notes_path=_LOCAL_PATHS["tiny_denials"],
            label_path=_LABEL_PATH,
            nrows=100,
        ),
        MultiNoteConfig(
            name="task2",
            version=VERSION,
            description="full dataset for task 2",
            note_type=["H&P"],
            label_map=task1_map,
            trunc_len=400,
            notes_path=_LOCAL_PATHS["claims_denials"],
            label_path=_LABEL_PATH,
            nrows=-1,  # keep full dataset
        ),
        MultiNoteConfig(
            name="task3_test",
            version=VERSION,
            description="debug dataset for task 3",
            note_type=["Discharge Narrative"],
            label_map=task3_map,
            trunc_len=400,
            notes_path=_LOCAL_PATHS["tiny_denials"],
            label_path=_LABEL_PATH,
            nrows=100,
        ),
        MultiNoteConfig(
            name="task3",
            version=VERSION,
            description="full dataset for task 3",
            note_type=["Discharge Narrative"],
            label_map=task3_map,
            trunc_len=400,
            notes_path=_LOCAL_PATHS["claims_denials"],
            label_path=_LABEL_PATH,
            nrows=-1,
        ),
        MultiNoteConfig(
            name="task4",
            version=VERSION,
            description="full dataset for task 4",
            note_type=["H&P"],
            label_map=task3_map,
            trunc_len=400,
            notes_path=_LOCAL_PATHS["claims_denials"],
            label_path=_LABEL_PATH,
            nrows=-1,
        ),
    ]

    DEFAULT_CONFIG_NAME = "debug"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        n_note_types = len(self.config.note_type)
        if n_note_types == 1:
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.Value("uint8"),
                }
            )

            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=features,
                homepage=_HOMEPAGE,
                license=_LICENSE,
                citation=_CITATION,
            )
        else:
            raise RuntimeError(
                f"dataset for {n_note_types} note types has not been implemented!"
            )

    def _split_generators(self, dl_manager):
        # doc: https://github.com/huggingface/datasets/blob/edf1902f954c5568daadebcd8754bdad44b02a85/src/datasets/builder.py#L1205
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        return [
            datasets.SplitGenerator(
                # These kwargs will be passed to _generate_examples
                name=self.config.name,
                gen_kwargs={
                    "trunc_len": self.config.trunc_len,
                    "notes_path": self.config.notes_path,
                    "label_path": self.config.label_path,
                    "label_map": self.config.label_map,
                    "note_type": self.config.note_type,
                    "nrows": self.config.nrows,
                },
            )
        ]

    # note: trunc_len not used. TODO: remove redundant arg
    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(
        self, trunc_len, notes_path, label_path, label_map, note_type, nrows
    ):
        # doc: https://github.com/huggingface/datasets/blob/edf1902f954c5568daadebcd8754bdad44b02a85/src/datasets/builder.py#L1216
        # can pass in any argument, as long as it's specified in gen_kwargs
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        # 1. iterate through the notes folder
        print("generating examples....")
        idx = 0
        label_df = pd.read_excel(_LABEL_PATH)
        for dirpath, dirnames, filenames in os.walk(notes_path):
            for subdir in dirnames:
                subdir_path = os.path.join(dirpath, subdir)
                for file in os.listdir(subdir_path):
                    filename = os.fsdecode(file)
                    file_type = filename.split("_")[0]
                    csn = filename.split("_")[-1]
                    # 2. Indexing the labels file for the corresponding single label (TODO: specify label type in init)
                    label_raw = label_df[label_df.CSN == int(csn)]["Denial Type"].item()
                    label = label_map(label_raw)
                    date = label_df.CSN == int(csn)["Enc - Admission Date"].item()
                    if label < 0:
                        continue
                    # SINGLE TYPE SINGLE LABEL Handling
                    # 3. for each encounter_csn, get the desired type of note. For empty note throw a warning
                    if file_type in note_type:
                        with open(os.path.join(subdir_path, filename)) as f:
                            note_content = f.read()
                            # 4. yield text and label
                            yield idx, {
                                "text": note_content,
                                "label": label,
                                "date": date,
                            }
                            idx = idx + 1
                            found = True
                            # option to keep only nrows
                            if nrows > 0 and idx >= nrows:
                                return 0
