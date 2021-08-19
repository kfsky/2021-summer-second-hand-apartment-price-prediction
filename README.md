# 2021-summer-second-hand-apartment-price-prediction
Nishika 主催 定期開催コンペ「中古マンション価格予測 2021夏の部

### 実験環境構成
mst8823さんの環境を参考に構成。
URL： https://github.com/mst8823/atmacup10

### directory 
```
├─input
├─notebooks
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
* LightGBM：CV=0.0714, LB=0.0729
* CatBoost：CV=0.0705, LB=0.0724



