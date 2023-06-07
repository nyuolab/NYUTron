# author: Grace (Ge'Er) Yang, Ming Cao
# modified by: Lavender Jiang

import pandas as pd
import numpy as np
import os
import xgboost
from sklearn import metrics
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime
import joblib
import wandb
import argparse
from omegaconf import OmegaConf
from evaluate import load

auc = load("roc_auc")
use_gpu = False


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print(
            "\n Time taken: %i hours %i minutes and %s seconds."
            % (thour, tmin, round(tsec, 2))
        )


def get_info(A):
    print(A)
    print(A.shape)


def preprocess(df, keep_cols, feats, categorical_feats, labels):
    df = df[keep_cols]
    for f in categorical_feats:
        df[f] = df[f].astype("category")
    # fill any missing value with average
    X = df[feats].fillna(df[feats].mean()).to_numpy()
    # get_info(X) # N by d
    Y = df[labels].to_numpy()
    # get_info(Y)
    return X, Y


def subsample(X, Y, n_samples):
    n_total = len(X)
    print(f"choosing {n_samples} out of {n_total}")
    sample_idx = rng.choice(np.arange(n_total), size=n_samples, replace=False)
    res_X = X[sample_idx]
    res_Y = Y[sample_idx]
    print(f"res X has shape {res_X.shape}, res Y has shape {res_Y.shape}")
    return res_X, res_Y


def evaluate(clf, X, Y, reg=True):
    if reg:
        pred = clf.predict(X)
    else:
        pred = clf.predict_proba(X)[:, 1]
    res = auc.compute(references=Y, prediction_scores=pred)["roc_auc"]
    return res


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="configs/readmission_toy.yaml"
    )
    parser.add_argument("--n_samples", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    seed = args.seed  # 36
    n_samples = args.n_samples  # 100
    conf = OmegaConf.load(args.config_path)
    entity = conf.entity
    # configure run
    njobs = 2  # 16
    folds = 3
    iters = 2  # 100
    rng = np.random.default_rng(seed)
    # initialize logging
    wandb.init(
        entity=entity,
        project=conf.project,
        name=f"{n_samples}samples_seed{seed}",
        config={"seed": seed, "n_samples": n_samples},
    )
    dataset_path = conf.dataset_path
    spatial_val = pd.read_csv(os.path.join(dataset_path, "val.csv"))
    # preprocess dataset to keep only relevant features
    keep_cols = OmegaConf.to_container(conf.keep_cols)
    feats = OmegaConf.to_container(conf.feats)
    categorical_feats = OmegaConf.to_container(conf.categorical_feats)
    if len(categorical_feats) > 0 and not use_gpu:
        raise RuntimeError(
            "categorical features are not supported on CPU, set use_gpu to True to proceed"
        )
    labels = OmegaConf.to_container(conf.labels)
    preprocess_func = lambda x: preprocess(
        x, keep_cols, feats, categorical_feats, labels
    )
    spatial_val_X, spatial_val_Y = preprocess_func(spatial_val)
    train = pd.read_csv(os.path.join(dataset_path, "train.csv"))
    train_X, train_Y = preprocess_func(train)
    # ## Combine train & val for cross validation
    train_val_X = np.concatenate([train_X, spatial_val_X])
    train_val_Y = np.concatenate([train_Y, spatial_val_Y])
    train_val_X, train_val_Y = subsample(train_val_X, train_val_Y, n_samples)

    # ## Train Random Forest Classifier using Spatial Val for Hyperparameter Search
    # Reference: https://www.datacamp.com/community/tutorials/xgboost-in-python
    # # From https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

    objectives = [
        "reg:linear",
        "reg:logistic",
        "reg:pseudohubererror",
        "reg:squarederror",
    ]
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    lambdas = [0.01, 0.1, 1]
    max_depths = [4, 5, 10, 15, 20]
    colsample_bytree = [0.3, 0.6, 1]
    n_estimators = [10, 100, 1000]

    # Create the grid
    param_grid = {
        "objective": objectives,
        "learning_rate": learning_rates,
        "lambda": lambdas,
        "max_depth": max_depths,
        "colsample_bytree": colsample_bytree,
    }
    pprint(param_grid)

    data_matrix = xgboost.DMatrix(data=train_X, label=train_Y)

    # reference: https://www.kaggle.com/code/tilii7/hyperparameter-grid-search-with-xgboost/notebook

    if use_gpu:
        xgb = xgboost.XGBClassifier(
            n_estimators=10,
            objective="binary:logistic",
            silent=True,
            nthread=1,
            enable_categorical=True,
            tree_method="gpu_hist",
        )
    else:
        xgb = xgboost.XGBClassifier(
            n_estimators=10, objective="binary:logistic", silent=True, nthread=1
        )
    params = {
        "min_child_weight": [1, 5, 10],
        "gamma": [0.5, 1, 1.5, 2, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "max_depth": [3, 4, 5],
        "learning_rate": learning_rates,
        "n_estimators": n_estimators,
    }
    print(f"random seed with seed {seed}")
    random_search = RandomizedSearchCV(
        xgb,
        param_distributions=params,
        n_iter=iters,
        scoring="roc_auc",
        cv=folds,
        verbose=10,
        random_state=seed,
        n_jobs=njobs,
    )
    # Here we go
    start_time = timer(None)  # timing starts from this point for "start_time" variable
    with joblib.parallel_backend(backend="loky", n_jobs=njobs):
        random_search.fit(train_val_X, train_val_Y)
    timer(start_time)  # timing ends here for "start_time" variable

    print(f"best param is {random_search.best_params_}")

    # ## Load Test Data
    spatial_test = pd.read_csv(os.path.join(dataset_path, "test.csv"))
    spatial_test_X, spatial_test_Y = preprocess_func(spatial_test)
    temporal_test = pd.read_csv(os.path.join(dataset_path, "temporal_test.csv"))
    temporal_test_X, temporal_test_Y = preprocess_func(temporal_test)
    # ## Eval on Test Data
    print("same time test result:")
    st_res = evaluate(random_search, spatial_test_X, spatial_test_Y, reg=False)
    print(st_res)
    print("temporal test result")
    tmp_res = evaluate(random_search, temporal_test_X, temporal_test_Y, reg=False)
    print(tmp_res)
    wandb.log({"temporal test auc": tmp_res, "same-time test auc": st_res})
