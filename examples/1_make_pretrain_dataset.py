import pandas as pd
import logging
from nyutron.data_utils import PretrainDataPipeline, DataProcessingModuleExchange


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

conf_path = "configs/pretrain_data_configs/toy_example.yaml"
data_opt = "small_synthetic_clinical"  # "toy_example"
pipe = PretrainDataPipeline(
    conf_path,
    tokenized_data_output_path=f"data/pretrain/tokenized_{data_opt}",
    tokenizer_output_path=f"data/pretrain/tokenizer_{data_opt}",
)
print(pipe)
if data_opt == "toy_example":
    df = pd.DataFrame(
        {"text": ["a.", "a. b.", "a. b. c.", "a. b. c. d."], "a": [1, 2, 3, 4]}
    )
    data = DataProcessingModuleExchange(data=df)
elif data_opt == "small_synthetic_clinical":
    path = "../synthetic_data/raw/synthetic_data.csv"
    data = DataProcessingModuleExchange(path=path)
res = pipe(data)
print(res.data)
print(res.path)
print(pipe)
pipe.print_history()
pipe.print_output_path()
