"""
xgbの各パラメータの説明
objective: 目的関数の指定です。回帰問題の場合は "reg:squarederror" を指定します。
eval_metric: 評価指標の指定です。回帰問題の場合は一般的に "rmse" (Root Mean Squared Error) を使用します。
max_depth: 決定木の深さの最大値です。過学習を防ぐために適切な値を設定します。
eta: 学習率です。各木の影響を制御します。小さな値を設定すると学習が安定しやすくなりますが、収束に時間がかかることもあります。
min_child_weight: 子ノードを作成するための最小の重み合計です。過学習を抑えるために用いられます。
subsample: 学習に使用するサンプルの割合です。0から1の間の値を指定します。1に近い値を設定するとオーバーフィッティングを抑えることができます。
colsample_bytree: 各木の特徴量の割合です。0から1の間の値を指定します。1に近い値を設定するとオーバーフィッティングを抑えることができます。
silent: 学習の進捗状況を表示するかどうかを指定します。0の場合は表示しません。
random_state: ランダムシードの指定です。再現性を保つために使用します。
num_round: 学習のイテレーション数です。木の数を指定します。
early_stopping_rounds: アーリーストッピングの設定です。指定したイテレーション数連続して検証データの評価指標が改善しなかった場合、学習を終了します。
"""

cf_params_xgb = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 9,
    "max_depth": 12,
    "eta": 0.1,
    "min_child_weight": 10,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "silent": 1,
    "random_state": 71,
    "num_round": 10000,
    "early_stopping_rounds": 10,
}

rg_params_xgb = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 12,
    "eta": 0.1,
    "min_child_weight": 10,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "silent": 1,
    "random_state": 71,
    "num_round": 10000,
    "early_stopping_rounds": 10,
}

features = [
    "Mannitol",
    "Peptone",
    "Yeast Extract",
    "Temperature",
    "Fermantation Time",
]

params_xgb_all = dict(cf_params_xgb)
params_xgb_all["num_round"] = 350

params_nn = {
    "layers": 3,
    # サンプルのため早く終わるように設定
    "nb_epoch": 5,  # 1000
    "patience": 10,
    "dropout": 0.5,
    "units": 512,
}
