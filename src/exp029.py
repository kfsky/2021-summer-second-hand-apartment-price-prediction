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
from mypipe.models.model_catboost import MyCatRegressor
from mypipe.Block_features import BaseBlock, ContinuousBlock, CountEncodingBlock, OheHotEncodingBlock, \
    LabelEncodingBlock, ArithmeticOperationBlock, AggregationBlock, WrapperBlock, AgeBlock


# ---------------------------------------------------------------------- #
# GPUの再現性確保を確認したいので、36行名を追加！
# 基本exp025と同じ
exp = "exp029"
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
    output_df["取引時点_年次"] = output_df["取引時点"].copy()
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

        del station_df
        gc.collect()

        return output_df[self.columns].copy()


class LandPriceBlock(BaseBlock):
    def __init__(self, column: str):
        self.column = column

    def transform(self, input_df):
        use_columns = [
            "市区町村コード",
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
            "Ｈ２０価格"]

        land_price_df = pd.read_csv("../add/gp_land_price.csv", usecols=use_columns)

        c = self.column
        output_df = pd.merge(input_df, land_price_df, on="市区町村コード", how="left")

        # memory
        del land_price_df
        gc.collect()

        return output_df[c]


class LandAggregationBlock(BaseBlock):
    def __init__(self, whole_df, key: str, agg_column: str, agg_funcs=["mean"], fillna=None):
        self.whole_df = whole_df
        self.key = key
        self.agg_column = agg_column
        self.agg_funcs = agg_funcs
        self.fillna = fillna

    def fit(self, input_df, y=None):
        use_columns = [
            "市区町村コード",
            "Ｒ３価格",
            "Ｒ２価格",
            "Ｈ３１価格",
            "Ｈ３０価格",
            "Ｈ２９価格",
        ]

        land_price_df = pd.read_csv("../add/gp_land_price.csv", usecols=use_columns)
        self.merge_df_ = pd.merge(self.whole_df, land_price_df, on="市区町村コード", how="left")

        del land_price_df
        gc.collect()

        return self.transform(input_df)

    def transform(self, input_df):
        if self.fillna:
            self.whole_df[self.agg_column] = self.whole_df[self.agg_column].fillna(self.fillna)

        self.group_df = self.merge_df_.groupby(self.key).agg({self.agg_column: self.agg_funcs}).reset_index()
        column_names = [f"LAND_GP_{self.agg_column}@{self.key}_{agg_func}" for agg_func in self.agg_funcs]

        self.group_df.columns = [self.key] + column_names
        output_df = pd.merge(input_df[self.key], self.group_df, on=self.key, how="left").drop(columns=[self.key])
        return output_df


# 入れるとスコアが下がる結果
def get_diff_price(input_df):
    use_columns = ["取引時点", "取引価格（総額）_log"]
    price_df = pd.read_csv("../input/train.csv", usecols=use_columns).groupby("取引時点")["取引価格（総額）_log"].mean().diff(
        2).reset_index()
    price_df.rename(columns={"取引時点": "取引時点_年次",
                             "取引価格（総額）_log": "price_diff"}, inplace=True)

    output_df = pd.merge(input_df, price_df, on="取引時点_年次", how="left")

    c = "price_diff"

    return output_df[c]


# 市町村情報を分割して生成
def get_municipalities(input_df):
    output_df = pd.DataFrame()
    output_df["郡"] = input_df["市区町村名"].str.contains("郡") * 1
    output_df["町"] = input_df["市区町村名"].str.endswith("町") * 1
    output_df["23区"] = ((input_df["都道府県名"] == "東京都") & (input_df["市区町村名"].str.contains("区"))) * 1

    return output_df


# LDKかどうかを判定
def get_ldk(input_df):
    output_df = pd.DataFrame()
    output_df["is_LDK"] = input_df["間取り"].str.contains("ＬＤＫ") * 1
    return output_df


# 緯度経度を取得する
class GeoCodeBlock(BaseBlock):
    def fit(self, input_df, y=None):
        return self.transform(input_df)

    def transform(self, input_df):
        geo_df = pd.read_csv("../add/geo_info.csv").rename(columns={"latitude": "Area_Latitude",
                                                                    "longitude": "Area_Longitude"})

        output_df = pd.merge(input_df, geo_df, on=["都道府県名", "市区町村名", "地区名"], how="left")

        return output_df[["Area_Latitude", "Area_Longitude"]]


# 不動産価格指数を追加する
class PriceIndexBlock(BaseBlock):
    def fit(self, input_df, y=None):
        return self.transform(input_df)

    def transform(self, input_df):
        price_index_df = pd.read_csv("../add/price_index.csv").groupby("取引時点").mean().reset_index()
        price_index_df.rename(columns={"取引時点": "取引時点_年次"}, inplace=True)
        output_df = pd.merge(input_df, price_index_df, on="取引時点_年次", how="left")

        columns = ["全国Japan季節調整"]

        return output_df[columns]


# aggregation の追加関数
def max_min(x):
    return x.max()-x.min()


def q75_q25(x):
    return x.quantile(0.75) - x.quantile(0.25)


class MeanPriceAggBlock(BaseBlock):
    def __init__(self, key, agg_funcs: str = ["mean"], split_year: int = None,):
        self.split_year = split_year
        self.key = key
        self.agg_funcs = agg_funcs

    def fit(self, input_df, y=None):
        if self.split_year is None:
            input_df["MeanPrice"] = (10 ** input_df["取引価格（総額）_log"]) / input_df["面積（㎡）"]
            self.target_map = input_df.groupby(self.key)["MeanPrice"].agg(self.agg_funcs)

        else:
            _input_df = input_df[input_df["取引時点"]>(2020-self.split_year)].copy()
            _input_df["MeanPrice"] = (10 ** input_df["取引価格（総額）_log"]) / input_df["面積（㎡）"]
            self.target_map = _input_df.groupby(self.key)["MeanPrice"].agg(self.agg_funcs)

        output_encoded = np.zeros((len(input_df), len(self.agg_funcs)))
        kf = KFold(n_splits=4, shuffle=True, random_state=1000)
        for index, agg_func in enumerate(self.agg_funcs):
            for train_idx, valid_idx in kf.split(input_df):
                target_map = input_df.iloc[train_idx].groupby(self.key)["MeanPrice"].agg(self.agg_funcs)
                output_encoded[valid_idx, index] = input_df.iloc[valid_idx][self.key].map(target_map[agg_func].to_dict())

        output_encoded = pd.DataFrame(output_encoded)
        if self.split_year is None:
            output_encoded.columns = [f'PRICE_TE_{self.key}_MeanPrice_{agg_func}' for agg_func in self.agg_funcs]
        else:
            output_encoded.columns = [f'PRICE_TE_{self.key}_MeanPrice_{self.split_year}years_{agg_func}' for agg_func in self.agg_funcs]

        return output_encoded

    def transform(self, input_df):
        output_encoded = np.zeros((len(input_df), len(self.agg_funcs)))
        for index, agg_func in enumerate(self.agg_funcs):
            output_encoded[:,index] = input_df[self.key].map(self.target_map['mean'].to_dict()).values

        output_encoded = pd.DataFrame(output_encoded)
        if self.split_year is None:
            output_encoded.columns = [f'PRICE_TE_{self.key}_MeanPrice_{agg_func}' for agg_func in self.agg_funcs]
        else:
            output_encoded.columns = [f'PRICE_TE_{self.key}_MeanPrice_{self.split_year}years_{agg_func}' for agg_func in self.agg_funcs]
        return output_encoded


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
        ArithmeticOperationBlock(target_column1='面積（㎡）', target_column2="建ぺい率（％）", operation="/"),
        ArithmeticOperationBlock(target_column1='面積（㎡）', target_column2="容積率（％）", operation="/"),
        *[AggregationBlock(whole_df=whole_df,
                           key="最寄駅：名称",
                           agg_column=c,
                           agg_funcs=["mean", "std", "max", "min", "count", max_min, q75_q25]) for c in ['最寄駅：距離（分）',
                                                                                                         '面積（㎡）',
                                                                                                         '建ぺい率（％）',
                                                                                                         '容積率（％）']],
        *[AggregationBlock(whole_df=whole_df,
                           key="都道府県名",
                           agg_column=c,
                           agg_funcs=["mean", "std", "max", "min", "count", max_min, q75_q25]) for c in ['最寄駅：距離（分）',
                                                                                                         '面積（㎡）',
                                                                                                         '建ぺい率（％）',
                                                                                                         '容積率（％）']],
        *[AggregationBlock(whole_df=whole_df,
                           key="地区名",
                           agg_column=c,
                           agg_funcs=["mean", "std", "max", "min", "count", max_min, q75_q25]) for c in ['最寄駅：距離（分）',
                                                                                                         '面積（㎡）',
                                                                                                         '建ぺい率（％）',
                                                                                                         '容積率（％）']],
        *[AggregationBlock(whole_df=whole_df,
                           key="市区町村名",
                           agg_column=c,
                           agg_funcs=["mean", "std", "max", "min", "count", max_min, q75_q25]) for c in ['最寄駅：距離（分）',
                                                                                                         '面積（㎡）',
                                                                                                         '建ぺい率（％）',
                                                                                                         '容積率（％）']],
        *[AggregationBlock(whole_df=whole_df,
                           key="最寄駅：距離（分）",
                           agg_column=c,
                           agg_funcs=["mean", "std", "max", "min", "count", max_min, q75_q25]) for c in ['面積（㎡）',
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
        *[StationPassengerBlock(columns=c) for c in ["乗降客数18",
                                                     "latitude",
                                                     "longitude"]],
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
        ]],
        *[LandAggregationBlock(whole_df=whole_df, key="最寄駅：名称", agg_column=c,
                               agg_funcs=["mean", "std"]) for c in [
            "Ｒ３価格",
            "Ｒ２価格",
            "Ｈ３１価格",
            "Ｈ３０価格",
            "Ｈ２９価格",
        ]],
        WrapperBlock(get_municipalities),
        WrapperBlock(get_ldk),
        GeoCodeBlock(),
        PriceIndexBlock(),
        *[MeanPriceAggBlock(key=c) for c in ["都道府県名", "市区町村名", "地区名", "最寄駅：名称"]],
        *[MeanPriceAggBlock(key=c, split_year=3) for c in ["都道府県名", "市区町村名", "地区名", "最寄駅：名称"]],
    ]

    # create train_x, train_y, test_x
    train_y = train["取引価格（総額）_log"]
    train_x = to_feature(train, process_blocks, is_train=True)
    test_x = to_feature(test, process_blocks)

    # delete train_df, test_df, release memory
    del train_df
    del test_df
    del whole_df
    del train
    del test
    gc.collect()

    # dump features
    joblib.dump(train_x, os.path.join("../output/" + exp + "/feature", "train_feat.pkl"), 3)
    joblib.dump(test_x, os.path.join("../output/" + exp + "/feature", "test_feat.pkl"), 3)

    # set model
    model = MyCatRegressor

    # set run params
    run_params = {
        "metrics": mean_absolute_error,
        "cv": make_skf,
        "feature_select_method": "tree_importance",
        "feature_select_fold": 5,
        "feature_select_num": 500,
        "folds": 5,
        "seeds": [71, 72, 73],
    }

    # set model params
    model_params = {
        "n_estimators": 20000,
        "objective": "MAE",
        "learning_rate": 0.01,
        "subsample": 0.8,
        "max_depth": 10,
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

