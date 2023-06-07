# author: Grace (Ge'Er) Yang, Ming Cao
# modified by: Lavender Jiang

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import evaluate, argparse, wandb


def from_csv(
    train_path, val_path, test_path, train_text_feature, test_text_feature, LABEL_COL
):
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    df_train = df_train[[train_text_feature, LABEL_COL]]
    df_val = df_val[[test_text_feature, LABEL_COL]]
    df_test = df_test[[test_text_feature, LABEL_COL]]
    return df_train, df_val, df_test


def xy_split(
    df_train, df_val, df_test, train_text_feature, test_text_feature, LABEL_COL
):
    X_train = df_train[train_text_feature]
    y_train = df_train[LABEL_COL]
    X_val = df_val[test_text_feature]
    y_val = df_val[LABEL_COL]
    X_test = df_test[test_text_feature]
    y_test = df_test[LABEL_COL]
    return X_train, y_train, X_val, y_val, X_test, y_test


def split_data(
    TRAIN_PATH, VAL_PATH, TEST_PATH, TRAIN_TEXT_FEATURE, TEST_TEXT_FEATURE, LABEL_COL
):
    df_train, df_val, df_test = from_csv(
        TRAIN_PATH,
        VAL_PATH,
        TEST_PATH,
        TRAIN_TEXT_FEATURE,
        TEST_TEXT_FEATURE,
        LABEL_COL,
    )
    return xy_split(
        df_train, df_val, df_test, TRAIN_TEXT_FEATURE, TEST_TEXT_FEATURE, LABEL_COL
    )


def transform_data(model, X):
    transformed = model[:-1].fit_transform(X)
    return transformed


def eval_data(model, X, y, metric):
    logits = model.predict_proba(X)
    pos_logits = logits[:, 1]
    results = metric.compute(references=y, prediction_scores=pos_logits)
    return results


def subsample(X, Y, n_samples, rng):
    n_total = len(X)
    print(f"choosing {n_samples} out of {n_total}")
    sample_idx = rng.choice(np.arange(n_total), size=n_samples, replace=False)
    res_X = X[sample_idx]
    res_Y = Y[sample_idx]
    print(f"res X has shape {res_X.shape}, res Y has shape {res_Y.shape}")
    return res_X, res_Y


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    config_defaults = {
        "max_depth": 3,
        "learning_rate": 0.04,
        "gamma": 9,
        "reg_alpha": 0,
        "reg_lambda": 4.5,
        "min_child_weight": 4,
        "colsample_bytree": 1,
        "scale_pos_weight": 16,
        "n_estimators": 100,
    }
    entity = "lavender"
    wandb.init(
        entity=entity, config=config_defaults
    )  # defaults are over-ridden during the sweep
    config = wandb.config
    TRAIN_PATH = "../examples/data/finetune/toy_readmission/4_way_splits/train.csv"
    VAL_PATH = "../examples/data/finetune/toy_readmission/4_way_splits/val.csv"
    TEST_PATH = "../examples/data/finetune/toy_readmission/4_way_splits/temporal_test.csv"  # temporal test
    SAME_TEST_PATH = "../examples/data/finetune/toy_readmission/4_way_splits/test.csv"  # same-time test
    TRAIN_TEXT_FEATURE = "text"
    TEST_TEXT_FEATURE = "text"
    LABEL_COL = "label"
    MAX_FEATURES = 10  # 5000 #512
    args = parser.parse_args()
    seed = args.seed
    n_samples = args.n_samples  # 100
    # configure run
    rng = np.random.default_rng(seed)
    # initialize logging
    # wandb.init(project='tf_idf_xgb_readmission_new', name=f'{n_samples}samples_seed{seed}_max{MAX_FEATURES}feats',
    #            config={'seed': seed, 'n_samples':n_samples, 'max_feat': MAX_FEATURES})
    model = Pipeline(
        [
            (
                "vect",
                CountVectorizer(
                    lowercase=True, max_features=MAX_FEATURES, dtype=np.float32
                ),
            ),
            ("tfidf", TfidfTransformer(use_idf=True, smooth_idf=True)),
            (
                "clf",
                XGBClassifier(
                    objective="binary:logistic",
                    n_jobs=10,
                    min_child_weight=config.min_child_weight,
                    reg_lambda=config.reg_lambda,
                    reg_alpha=config.reg_alpha,
                    colsample_bytree=config.colsample_bytree,
                    scale_pos_weight=config.scale_pos_weight,
                    learning_rate=config.learning_rate,  # .04,
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    gamma=config.gamma,
                    verbosity=3,
                    seed=seed,
                ),
            ),
        ]
    )
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        TRAIN_PATH,
        VAL_PATH,
        TEST_PATH,
        TRAIN_TEXT_FEATURE,
        TEST_TEXT_FEATURE,
        LABEL_COL,
    )
    X_train, y_train = subsample(X_train, y_train, n_samples, rng=rng)
    X_train_transformed = transform_data(model, X_train)
    x_val_transformed = transform_data(model, X_val)
    # train model
    model[-1].fit(
        X=X_train_transformed,
        y=y_train,
        eval_metric="auc",
        eval_set=[(x_val_transformed, y_val)],
        early_stopping_rounds=30,
    )  # 30,)
    # evaluate model
    roc_auc_score = evaluate.load("roc_auc")
    print("temporal test result:")
    tmp_res = eval_data(model, X_test, y_test, roc_auc_score)
    print(tmp_res)
    print("same time test result:")
    df_st = pd.read_csv(SAME_TEST_PATH)
    st_test_x = df_st[TEST_TEXT_FEATURE]
    st_test_y = df_st[LABEL_COL]
    st_res = eval_data(model, st_test_x, st_test_y, roc_auc_score)
    print(st_res)
    train_res = eval_data(model, X_train, y_train, roc_auc_score)
    print(train_res)
    val_res = eval_data(model, X_val, y_val, roc_auc_score)
    wandb.log(
        {
            "temporal test auc": tmp_res["roc_auc"],
            "same-time test auc": st_res["roc_auc"],
            "train-set auc": train_res["roc_auc"],
            "val-set auc": val_res["roc_auc"],
        }
    )


if __name__ == "__main__":
    sweep_config = {
        "method": "random",
        "metric": {"name": "val-set auc", "goal": "maximize"},
        "parameters": {
            "max_depth": {"values": [3, 6, 9, 12]},
            "learning_rate": {"values": [0.1, 0.05, 0.2]},
            "gamma": {"values": [1, 5, 9]},
            "min_child_weight": {"values": [0, 2, 4, 8]},
            "reg_lambda": {"values": [1, 5, 20]},
            "reg_alpha": {"values": [0, 0.5, 1]},
            "colsample_bytree": {"values": [0.1, 0.5, 1]},
            "scale_pos_weight": {"values": [1, 5, 10, 15]},
            "n_estimators": {"values": [100, 500, 1000]},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="TFIDF-XGBoost-readmission-sweeps")

    wandb.agent(sweep_id, train, count=10)
