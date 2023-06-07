import pandas as pd
import numpy as np

task = "comorbidity"  #'readmission'

seed = 0
temporal_split = True
rng = np.random.default_rng(seed=seed)
path = "synthetic_data.csv" if temporal_split else "synthetic_data_larger.csv"
df = pd.read_csv(path, index_col=0)
print(df)
if task == "readmission":
    labels = rng.integers(low=0, high=2, size=len(df))
elif task == "comorbidity":
    labels = rng.integers(low=0, high=6, size=len(df))

    def map_quantile_new(x):
        quantile_1 = 0
        quantile_2 = 2
        quantile_3 = 4
        if x <= quantile_1:
            res = 0
        elif x <= quantile_2:
            res = 1
        elif x <= quantile_3:
            res = 2
        else:
            res = 3
        return res

    labels = [map_quantile_new(x) for x in labels]
else:
    raise ValueError(f"Unknown task: {task}")
df[task] = labels
print(df)
if temporal_split:
    df.to_csv(f"synthetic_{task}_temporal.csv")
else:
    df.to_csv(f"synthetic_{task}_larger_w_labels.csv")
