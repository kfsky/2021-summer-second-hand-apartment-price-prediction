import os
import warnings
import sys
import joblib
import re

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from time import time
from contextlib import contextmanager
import gc

sys.path.append("../")

from mypipe.config import Config
from mypipe.utils import reduce_mem_usage
from mypipe.experiment import exp_env
from mypipe.experiment.runner import Runner
from mypipe.models.model_ridge import MyRidgeModel
from mypipe.utils import Util


# ---------------------------------------------------------------------- #
# stacking_final
exp = "exp066"
config = Config(EXP_NAME=exp, TARGET="PRICE")
exp_env.make_env(config)
rcParams['font.family'] = 'Noto Sans CJK JP'
os.environ["PYTHONHASHSEED"] = "0"
# ---------------------------------------------------------------------- #


@contextmanager
def timer(logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None):
    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time()
    yield
    d = time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)


# make KFold
def make_kf(train_x, train_y, n_splits, random_state=71):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(train_x, train_y))


# make stratified KFold
# 目的変数に対して、StratifiedKFoldを行っている関数
def make_skf(train_x, train_y, n_splits, random_state=71):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    s = 10
    _y = pd.cut(train_y, s, labels=range(s))
    return list(skf.split(train_x, _y))


# plot result
def result_plot(train_y, oof):
    name = "result"
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.distplot(train_y, label='train_y', color='orange')
    sns.distplot(oof, label='oof')
    ax.legend()
    ax.grid()
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(os.path.join(config.REPORTS, f'{name}.png'), dpi=120)  # save figure
    plt.show()


# create submission
def create_submission(preds):
    sample_sub = pd.read_csv(os.path.join(config.INPUT, "sample_submission.csv"))
    post_preds = [0 if x < 0 else x for x in preds]
    sample_sub["取引価格（総額）_log"] = post_preds
    sample_sub.to_csv(os.path.join(config.SUBMISSION, f'{config.EXP_NAME}.csv'), index=False)


# ---------------------------------------------------------------------- #
def main():
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    train = pd.read_csv(os.path.join(config.INPUT, "train.csv"))

    # read oof and pred
    exp_list = ["exp043", "exp044", "exp045", "exp046", "exp047", "exp048", "exp049", "exp050", "exp051", "exp052",
                "exp053", "exp054", "exp061", "exp062", "exp063", "exp064"]
    train_x, test_x = pd.DataFrame(), pd.DataFrame()
    for exp in exp_list:
        train_x[exp] = Util.load(f"../output/{exp}/preds/oof_PRICE.pkl")
        test_x[exp] = Util.load(f"../output/{exp}/preds/preds_PRICE.pkl")

    # create train_x, train_y, test_x
    train_y = train["取引価格（総額）_log"]



    # dump features
    joblib.dump(train_x, os.path.join("../output/" + exp + "/feature", "train_feat.pkl"), 3)
    joblib.dump(test_x, os.path.join("../output/" + exp + "/feature", "test_feat.pkl"), 3)

    # set model
    model = MyRidgeModel

    # set run params
    run_params = {
        "metrics": mean_absolute_error,
        "cv": make_kf,
        "folds": 5,
        "feature_select_method": None,
        "feature_select_fold": None,
        "feature_select_num": 50,
        "seeds": [71, 72, 73, 74, 75],
    }

    # set model params
    model_params = {
        "random_state": 2021
    }

    # fit params
    fit_params = {
        #"early_stopping_rounds": 100,
        #"verbose": 1000
    }

    # features
    features = {
        "train_x": train_x,
        "test_x": test_x,
        "train_y": train_y
    }

    # run model
    config.RUN_NAME = f'_{config.TARGET}'
    runner = Runner(config=config,
                    run_params=run_params,
                    model_params=model_params,
                    fit_params=fit_params,
                    model=model,
                    features=features,
                    use_mlflow=False
                    )
    runner.run_train_cv()
    runner.run_predict_cv()

    # make_submission
    create_submission(preds=runner.preds)

    # plot result
    result_plot(train_y=train_y, oof=runner.oof)


if __name__ == "__main__":
    main()

