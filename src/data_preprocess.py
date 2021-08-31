import pandas as pd
import numpy as np
import re
from functools import partial
from glob import glob

import openpyxl
import csv
import datetime

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


def merge_train_data():
    paths = glob("../train/*")
    train_dfs = []

    for path in paths:
        train_df = pd.read_csv(path)
        train_dfs.append(train_df)

    train_df = pd.concat(train_dfs)
    train_df.reset_index(drop=True, inplace=True)

    train_df.to_csv("../input/train.csv", index=False)


def get_landprince():
    price_df = pd.read_csv("../add/landprice.csv", encoding="cp932")
    gp_df = price_df.groupby("所在地コード").mean().reset_index().rename(columns={"所在地コード": "市区町村コード"})
    gp_df.to_csv("../add/gp_land_price.csv", index=False)


def convert_torihiki(x, y):
    if y in [4,5,6]:
        return str(x)+"年第１四半期"
    elif y in [7,8,9]:
        return str(x)+"年第２四半期"
    elif y in [10, 11, 12]:
        return str(x)+"年第３四半期"
    else:
        return str(x)+"年第４四半期"


def get_price_index_data():
    wb = openpyxl.load_workbook("../add/price_index.xlsx")
    get_sheets = ['全国Japan季節調整',
                  '北海道地方Hokkaido季節調整',
                  '東北地方Tohoku季節調整',
                  '関東地方Kanto季節調整',
                  '北陸地方Hokuriku季節調整',
                  '中部地方Chubu季節調整',
                  '近畿地方Kinki季節調整',
                  '中国地方Chugoku季節調整',
                  '四国地方Shikoku季節調整',
                  '九州・沖縄地方Kyushu-Okinawa季節調整', ]

    total_data = {}

    for sheet in get_sheets:
        values = []
        current_sheet = wb[sheet]
        for row in current_sheet["K10:K162"]:
            for col in row:
                values.append(col.value)

        total_data[sheet] = values

    data_list = []
    index_sheet = wb["全国Japan季節調整"]
    for row in index_sheet["A10:A162"]:
        for col in row:
            data_list.append(col.value)

    output_df = pd.DataFrame(total_data)
    output_df["年月"] = data_list
    output_df["年"] = output_df["年月"].dt.year
    output_df["月"] = output_df["年月"].dt.month
    output_df["取引時点"] = output_df[["年", "月"]].apply(lambda x: convert_torihiki(x[0], x[1]), axis=1)

    output_df = output_df.drop(["年月", "年", "月"], axis=1)

    output_df.to_csv("../add/price_index.csv", index=False)


def get_geo_data():
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")

    whole_df = pd.concat([train_df, test_df], axis=0)
    whole_df["所在地"] = whole_df["都道府県名"] + whole_df["市区町村名"] + whole_df["地区名"]
    geo_df = whole_df.groupby("所在地").count().reset_index()[["所在地", "ID"]]

    geolocator = Nominatim(user_agent="test1")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    geo_df['location'] = geo_df['所在地'].apply(geocode, timeout=30000)

    geo_df['latitude'] = geo_df['location'].apply(lambda x: x.point[0] if x else None)
    geo_df['longitude'] = geo_df['location'].apply(lambda x: x.point[1] if x else None)

    geo_df = geo_df.drop(["ID", "location"], axis=1)
    merge_df = pd.merge(geo_df, whole_df[["都道府県名",
                                          "市区町村名",
                                          "地区名",
                                          "所在地"]], on="所在地", how="left").drop_duplicates()
    merge_df.to_csv("../add/geo_info.csv", index=False)
