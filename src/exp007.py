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
from sklearn.model_selection import KFold
from tqdm import tqdm
from time import time
from contextlib import contextmanager

sys.path.append("../")

from mypipe.config import Config
from mypipe.utils import reduce_mem_usage
from mypipe.experiment import exp_env
from mypipe.experiment.runner import Runner
from mypipe.models.model_lgbm import MyLGBMModel
from mypipe.Block_features import BaseBlock, ContinuousBlock, CountEncodingBlock, OheHotEncodingBlock, \
    LabelEncodingBlock, ArithmeticOperationBlock, AggregationBlock, WrapperBlock, AgeBlock


# ---------------------------------------------------------------------- #
exp = "exp007"
config = Config(EXP_NAME=exp, TARGET="PRICE")
exp_env.make_env(config)
rcParams['font.family'] = 'Noto Sans CJK JP'
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


def get_function(block, is_train):
    s = mapping = {
        True: 'fit',
        False: 'transform'
    }.get(is_train)
    return getattr(block, s)


def to_feature(input_df,
               blocks,
               is_train=False):
    out_df = pd.DataFrame()

    for block in tqdm(blocks, total=len(blocks)):
        func = get_function(block, is_train)

        with timer(prefix='create ' + str(block) + ' '):
            _df = func(input_df)
        assert len(_df) == len(input_df), func.__name__
        out_df = pd.concat([out_df, _df], axis=1)
    return reduce_mem_usage(out_df)


# make KFold
def make_kf(train_x, train_y, n_splits, random_state=71):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(train_x, train_y))


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


# 最寄り情報の前処理関数
def normalize_moyori(moyori):
    if moyori == moyori:
        if moyori == '30分?60分':
            moyori = 45
        elif moyori == '1H?1H30':
            moyori = 75
        elif moyori == '1H30?2H':
            moyori = 105
        elif moyori == '2H?':
            moyori = 120
        moyori = int(moyori)
    return moyori


# 面積情報の前処理関数
def normalize_area(area):
    if area == area:
        area = int(re.sub('m\^2未満|㎡以上', '', str(area)))
    return area


# 西暦情報に変換する関数
def convert_wareki_to_seireki(wareki):
    if wareki == wareki:
        if wareki == '戦前':
            wareki = '昭和20年'
        value = wareki[2:-1]
        if value == '元':
            value = 1
        else:
            value = int(value)
        if '昭和' in wareki:
            seireki = 1925+value
        elif '平成' in wareki:
            seireki = 1988+value
        elif '令和' in wareki:
            seireki = 2018+value
    else:
        seireki = wareki
    return seireki


# preprocess
def preprocess(input_df):
    output_df = input_df.copy()

    output_df['最寄駅：距離（分）'] = output_df['最寄駅：距離（分）'].apply(lambda x: normalize_moyori(x))
    output_df['面積（㎡）'] = output_df['面積（㎡）'].apply(lambda x: normalize_area(x))
    output_df['建築年'] = output_df['建築年'].apply(lambda x: convert_wareki_to_seireki(x))

    # 建ぺい率と容積率を変換
    output_df["建ぺい率（％）"] = output_df["建ぺい率（％）"] / 100
    output_df["容積率（％）"] = output_df["容積率（％）"] / 100

    # 取引時点の年数を算出
    output_df["取引時点"] = output_df['取引時点'].apply(lambda x: int(x[:4]))

    return output_df


def get_rooms(input_df):
    output_df = pd.DataFrame()
    output_df['面積（㎡）'] = input_df['面積（㎡）'].apply(lambda x: normalize_area(x))

    output_df["L"] = input_df["間取り"].map(lambda x: 1 if "L" in str(x) else 0)
    output_df["D"] = input_df["間取り"].map(lambda x: 1 if "D" in str(x) else 0)
    output_df["K"] = input_df["間取り"].map(lambda x: 1 if "K" in str(x) else 0)
    output_df["R"] = input_df["間取り"].map(lambda x: 1 if "R" in str(x) else 0)
    output_df["S"] = input_df["間取り"].map(lambda x: 1 if "S" in str(x) else 0)

    output_df["Maisonette"] = input_df["間取り"].map(lambda x: 1 if "メゾネット" in str(x) else 0)
    output_df["OpenFloor"] = input_df["間取り"].map(lambda x: 1 if "オープンフロア" in str(x) else 0)
    output_df["Studio"] = input_df["間取り"].map(lambda x: 1 if "スタジオ" in str(x) else 0)

    output_df["Special"] = output_df["Maisonette"] + output_df["OpenFloor"] + output_df["Studio"]
    output_df["RoomNum"] = input_df['間取り'].map(lambda x: re.sub("\\D", "", str(x)))
    output_df["RoomNum"] = output_df["RoomNum"].map(lambda x: int(x) if x != "" else 0)

    output_df["TotalRoomNum"] = output_df[['L', 'D', 'K', 'S', 'R', 'RoomNum']].sum(axis=1)
    output_df["RoomNumRatio"] = output_df["RoomNum"] / output_df["TotalRoomNum"]

    output_df["area_per_room"] = output_df['面積（㎡）'] / output_df["TotalRoomNum"]
    output_df = output_df.drop('面積（㎡）', axis=1)

    return output_df


def get_distance(input_df):
    output_df = pd.DataFrame()
    output_df["distance"] = input_df["最寄駅：距離（分）"] * 80
    return output_df


class TargetEncodingBlock(BaseBlock):
    def __init__(self, column: str, target_column: str = "取引価格（総額）_log", agg_funcs: str = ["mean"]):
        self.column = column
        self.target_column = target_column
        self.agg_funcs = agg_funcs

    def fit(self, input_df, y=None):
        self.target_map = input_df.groupby(self.column)[self.target_column].agg(self.agg_funcs)

        output_encoded = np.zeros((len(input_df), len(self.agg_funcs)))
        kf = KFold(n_splits=5, shuffle=True, random_state=2021)
        for index, agg_func in enumerate(self.agg_funcs):
            for train_idx, valid_idx in kf.split(input_df):
                target_map = input_df.iloc[train_idx].groupby(self.column)[self.target_column].agg(self.agg_funcs)
                output_encoded[valid_idx, index] = input_df.iloc[valid_idx][self.column].map(target_map[agg_func].to_dict())

        output_encoded = pd.DataFrame(output_encoded)
        output_encoded.columns = [f'TE_{self.column}_{agg_func}' for agg_func in self.agg_funcs]
        return output_encoded

    def transform(self, input_df):
        output_encoded = np.zeros((len(input_df), len(self.agg_funcs)))
        for index, agg_func in enumerate(self.agg_funcs):
            output_encoded[:, index] = input_df[self.column].map(self.target_map['mean'].to_dict()).values

        output_encoded = pd.DataFrame(output_encoded)
        output_encoded.columns = [f'TE_{self.column}_{agg_func}' for agg_func in self.agg_funcs]

        return output_encoded


class StationPassengerBlock(BaseBlock):
    def __init__(self, columns: str):
        self.columns = columns

    def transform(self, input_df):
        station_df = pd.read_csv("../add/station.csv").drop_duplicates("最寄駅：名称")
        output_df = pd.merge(input_df, station_df, on="最寄駅：名称", how="left")

        return output_df[self.columns].copy()


class LandPriceBlock(BaseBlock):
    def __init__(self, column: str):
        self.column = column

    def transform(self, input_df):
        land_price_df = pd.read_csv("../add/landprice.csv", encoding="cp932").groupby("所在地コード").mean().reset_index()
        land_price_df.rename(columns={"所在地コード": "市区町村コード"}, inplace=True)

        c = self.column

        output_df = pd.merge(input_df, land_price_df, on="市区町村コード", how="left")

        return output_df[c]


# ---------------------------------------------------------------------- #
def main():
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    train_df = pd.read_csv(os.path.join(config.INPUT, "train.csv"))
    test_df = pd.read_csv(os.path.join(config.INPUT, "test.csv"))

    # 前処理・異常値処理
    train = preprocess(train_df)
    test = preprocess(test_df)

    whole_df = pd.concat([train, test], axis=0)

    process_blocks = [
        *[ContinuousBlock(c) for c in [
            "建ぺい率（％）",
            "容積率（％）",
            '面積（㎡）',
            '最寄駅：距離（分）',
            '建築年',
            "市区町村コード",
            '取引時点'
        ]],
        *[CountEncodingBlock(c, whole_df=whole_df) for c in [
            "都道府県名",
            "市区町村名",
            "地区名",
            "最寄駅：名称",
            "間取り",
            "建物の構造",
            "用途",
            "今後の利用目的",
            "都市計画",
            "改装",
            "取引の事情等",
        ]],
        *[LabelEncodingBlock(c, whole_df=whole_df) for c in [
            "都道府県名",
            "市区町村名",
            "地区名",
            "最寄駅：名称",
            "間取り",
            "建物の構造",
            "用途",
            "今後の利用目的",
            "都市計画",
            "改装",
            "取引の事情等"
        ]],
        *[AgeBlock(c) for c in [
            '建築年',
            "取引時点"
        ]],
        ArithmeticOperationBlock(target_column1='面積（㎡）', target_column2="建ぺい率（％）", operation="*"),
        ArithmeticOperationBlock(target_column1='面積（㎡）', target_column2="容積率（％）", operation="*"),
        *[AggregationBlock(whole_df=whole_df,
                           key="最寄駅：名称",
                           agg_column=c ,
                           agg_funcs=["mean", "std", "max", "min"]) for c in ['最寄駅：距離（分）',
                                                                              '面積（㎡）',
                                                                              '建ぺい率（％）',
                                                                              '容積率（％）']],
        *[AggregationBlock(whole_df=whole_df,
                           key="地区名",
                           agg_column=c,
                           agg_funcs=["mean", "std", "max", "min"]) for c in ['最寄駅：距離（分）',
                                                                              '面積（㎡）',
                                                                              '建ぺい率（％）',
                                                                              '容積率（％）']],
        WrapperBlock(get_rooms),
        *[TargetEncodingBlock(column=c, target_column="取引価格（総額）_log") for c in [
            "都道府県名",
            "市区町村名",
            "地区名",
            "最寄駅：名称",
            "間取り",
            "建物の構造",
            "用途",
            "今後の利用目的",
            "都市計画",
            "改装",
            "取引の事情等"
        ]],
        StationPassengerBlock(columns="乗降客数18"),
        WrapperBlock(get_distance),
        *[LandPriceBlock(column=c) for c in [
            "Ｒ３価格",
            "Ｒ２価格",
            "Ｈ３１価格",
            "Ｈ３０価格",
            "Ｈ２９価格",
            "Ｈ２８価格",
            "Ｈ２７価格",
            "Ｈ２６価格",
            "Ｈ２５価格",
            "Ｈ２４価格",
            "Ｈ２３価格",
            "Ｈ２２価格",
            "Ｈ２１価格",
            "Ｈ２０価格",
        ]]
    ]

    # create train_x, train_y, test_x
    train_y = train["取引価格（総額）_log"]
    train_x = to_feature(train, process_blocks, is_train=True)
    test_x = to_feature(test, process_blocks)

    # dump features
    joblib.dump(train_x, os.path.join("../output/" + exp + "/feature", "train_feat.pkl"))
    joblib.dump(test_x, os.path.join("../output/" + exp + "/feature", "test_feat.pkl"))

    # set model
    model = MyLGBMModel

    # set run params
    run_params = {
        "metrics": mean_absolute_error,
        "cv": make_kf,
        "feature_select_method": "tree_importance",
        "feature_select_fold": 5,
        "feature_select_num": 500,
        "folds": 5,
        "seeds": [71, 72, 73],
    }

    # set model params
    model_params = {
        "n_estimators": 20000,
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.01,
        "num_leaves": 1024,
        "n_jobs": -1,
        "importance_type": "gain",
        "reg_lambda": .5,
        "colsample_bytree": .5,
        "max_depth": 7,
    }

    # fit params
    fit_params = {
        "early_stopping_rounds": 100,
        "verbose": 1000
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

