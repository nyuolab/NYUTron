import pandas as pd
import logging
from nyutron.data_utils import FinetuneDataPipeline, DataProcessingModuleExchange


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

data_opt = "toy_readmission"  # "toy_comorbidity"
conf_path = (
    "configs/finetune_data_configs/toy_example.yaml"
    if not data_opt == "toy_comorbidity"
    else "configs/finetune_data_configs/toy_comorbidity.yaml"
)
pipe = FinetuneDataPipeline(
    conf_path,
    tokenized_data_output_path=f"data/finetune/{data_opt}/tokenized",
    split_data_output_path=f"data/finetune/{data_opt}/4_way_splits",
)
print(pipe)
if data_opt == "toy_example":
    df = pd.DataFrame(
        {
            "text": ["a.", "a. b.", "a. b. c.", "a. b. c. d."],
            "readmission": [1, 0, 1, 0],
        }
    )
    temporal_df = pd.DataFrame(
        {"text": ["c.", "d.", "e. f."], "readmission": [0, 0, 1]}
    )
    data = DataProcessingModuleExchange(data=df)
    temporal_test = DataProcessingModuleExchange(data=temporal_df)
elif data_opt == "toy_readmission":
    path = "../synthetic_data/raw/synthetic_readmission_larger_w_labels.csv"
    temporal_path = "../synthetic_data/raw/synthetic_readmission_temporal.csv"
    data = DataProcessingModuleExchange(path=path)
    temporal_test = DataProcessingModuleExchange(path=temporal_path)
elif data_opt == "toy_comorbidity":
    path = "../synthetic_data/raw/synthetic_comorbidity_larger_w_labels.csv"
    temporal_path = "../synthetic_data/raw/synthetic_comorbidity_temporal.csv"
    data = DataProcessingModuleExchange(path=path)
    temporal_test = DataProcessingModuleExchange(path=temporal_path)
else:
    raise NotImplementedError(f"data option {data_opt} not implemented!")
res = pipe(data, temporal_test=temporal_test)
print(res.data)
print(res.path)
print(pipe)
pipe.print_history()
pipe.print_output_path()
