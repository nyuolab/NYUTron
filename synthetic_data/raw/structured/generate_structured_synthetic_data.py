import pandas as pd
import numpy as np

task = "readmission"
struct_names = ["length_of_stay", "acute_visit", "past_ed_visit", "charles_comorbidity"]
struct_ranges = [(0, 30), (0, 1), (0, 3), (0, 5)]
seed = 0
temporal_split = False  # True
rng = np.random.default_rng(seed=seed)
path = (
    "../synthetic_readmission_temporal.csv"
    if temporal_split
    else "../synthetic_readmission_larger_w_labels.csv"
)
df = pd.read_csv(path, index_col=0)
print(df)
for struct_name, struct_range in zip(struct_names, struct_ranges):
    feat = rng.integers(low=struct_range[0], high=struct_range[1] + 1, size=len(df))
    df[struct_name] = feat
print(df)
if temporal_split:
    df.to_csv(f"synthetic_{task}_temporal.csv")
else:
    df.to_csv(f"synthetic_{task}_larger_w_labels.csv")
