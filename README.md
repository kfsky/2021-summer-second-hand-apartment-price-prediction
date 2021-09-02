# 2021-summer-second-hand-apartment-price-prediction
Nishika 主催 定期開催コンペ「中古マンション価格予測 2021夏の部

### Solution OverView
以下のような構成

### Machine Spec
CPU: Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz  
GPU: NVIDIA GeForce RTX2070 with Max-Q (8GB GDDR6 VRAM)  
RAM: 16GB

### Requirement
* Docker
* Docker-compose

メモリ不足になる場合があるので、自身のPCではswap拡張を実施
```
sudo fallocate -l 10G /swapfile2
sudo chmod 600 /swapfile2
sudo mkswap /swapfile2
sudo swapon /swapfile2
```

### 実験環境構成
mst8823さんの環境を参考に構成。
URL： https://github.com/mst8823/atmacup10

### directory 
```
├─input
├─notebooks
├─add
├─train
├─output
│  └─exp0XXX
│      ├─cols
│      ├─feature
│      ├─preds
│      ├─reports
│      └─trained
├─scr
│  └─mypipe
│    ├─experiment
│    ├─features
│    └─models
└─submission
```

* output は実験スクリプトを走らせると動的に作成される。
* src ディレクリ内に実験スクリプトを作成する必要がある。
* 実験管理は1実験1スクリプト

### run experiment
追加データ作成は以下の.ipynbファイルを実行する。(notebookファイルを上から実行すればいい)
* merge_train.ipynb （trainにあるcsvファイルをmergeする）
* get_geo.ipynb （地理情報データの加工）
* convert_eki.ipynb (駅情報データの加工)
* get_price_index.ipynb （住宅価格指数データの加工）
* land_price.ipynb （公示地価データの加工）

```
cd 2021-summer-second-hand-apartment-price-prediction/src
python exp0xx.py
```

### 1. feature engineering
1. LabelEncoding, CountEncoding, TargetEncodingを使用
2. 集約特徴量を作成  
   チュートリアルの集約系に追加していく形で特徴量を増加させていった。
3. 建築年など新規特徴量の作成
4. 外部データを使用   
   不動産価格指数（国交省）：https://www.mlit.go.jp/totikensangyo/totikensangyo_tk5_000085.html  
   駅別乗降者データ：https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-S12-v2_6.html  
   公示地価：https://nlftp.mlit.go.jp/ksj/old/datalist/old_KsjTmplt-L01.html
   
### 2. cv
* StratifiedKFold (n=5)

### 3. model
#### 1. Single model
* LightGBM：CV=0.0713, LB=0.0728
* CatBoost：CV=0.0705, LB=0.0724  

LightGBM: 7model, CatBoost: 3modelをStackingに使用

#### 2. Stacking_1st
* Ridge * 2, LightGBM * 2, CatBoost * 2, XGBoost * 1, MLP * 3
* CV = 5 or 10
* Best: CV=0.0700, LB=0.0714 (Ridge, XGBoost, MLP)

#### 3. Stacking_2nd
Single model, stacking_1stをすべて使って再度学習
* Ridge, LightGBM, XGBoost
* CV = 5
* Best: CV=0.0699, LB=0.0713(Ridge)

#### 4. final layer
Stacking 2ndの結果のaveraging
