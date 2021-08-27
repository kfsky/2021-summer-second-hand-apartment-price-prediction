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
from mypipe.models.model_mlp import MyMLPModel
from mypipe.utils import Util


# ---------------------------------------------------------------------- #
# stacking_1st
exp = "exp049"
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
    import tensorflow as tf
    from tensorflow.keras import layers as L
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    train = pd.read_csv(os.path.join(config.INPUT, "train.csv"))

    # read oof and pred
    exp_list = ["exp031", "exp032", "exp033", "exp034", "exp036", "exp037", "exp038", "exp039", "exp040", "exp042"]
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
    model = MyMLPModel

    # set run params
    run_params = {
        "metrics": mean_absolute_error,
        "cv": make_skf,
        "folds": 5,
        "feature_select_method": None,
        "feature_select_fold": None,
        "feature_select_num": 10,
        "seeds": [28, 29, 30, 31, 32]
    }

    # MLPの設定を行う
    def build_model(input_dim, output_dim):
        inp = L.Input(shape=(input_dim,))

        x = L.Dense(2 ** 9)(inp)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        x = L.Dropout(0.2)(x)

        out = L.Dense(output_dim)(x)
        mpl_model = tf.keras.Model(inputs=inp, outputs=out)
        mpl_model.compile(optimizer="adam", loss="mse")

        return mpl_model

    def set_callbacks():
        early_stopping = EarlyStopping("val_loss", patience=4, verbose=3, baseline=None, restore_best_weights=True)
        reduce_lr_loss = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1, mode="min")
        return [reduce_lr_loss, early_stopping]

    # set model params
    model_params = {
        "model": build_model,
        "scale": True,
        "multiple_target": False
    }

    # fit params
    fit_params = {
        "epochs": 50,
        "batch_size": 128,
        "callbacks": set_callbacks(),
        "verbose": 1
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

