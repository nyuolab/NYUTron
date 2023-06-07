import pandas as pd
import os

train_val_split_path = (
    "../synthetic_data/raw/structured/synthetic_readmission_larger_w_labels.csv"
)
temporal_path = "../synthetic_data/raw/structured/synthetic_readmission_temporal.csv"
reference_split_path = "../examples/data/finetune/toy_readmission/4_way_splits"
output_path = "../synthetic_data/raw/structured/toy_readmission_splits"


train_val_split = pd.read_csv(train_val_split_path)
for split in ["train", "val", "test"]:
    df = pd.read_csv(os.path.join(reference_split_path, f"{split}.csv")).rename(
        columns={"label": "readmission"}
    )
    print(df)
    struct_df = pd.merge(
        left=df, right=train_val_split, on=["text", "readmission"], how="inner"
    )
    print(struct_df)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    struct_df.to_csv(os.path.join(output_path, f"{split}.csv"))
temporal_split = pd.read_csv(temporal_path)
temporal_split.to_csv(os.path.join(output_path, "temporal_test.csv"))
