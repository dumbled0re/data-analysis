from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from model import Model
from utils.logger import Logger
from utils.util import Util

logger = Logger()


class Runner:
    def __init__(
        self,
        run_name: str,
        model_cls: Callable[[str, dict], Model],
        features: List[str],
        params: dict,
        x_train_path: str,
        y_train_path: str,
        x_test_path: str,
    ):
        """コンストラクタ

        :param run_name: ランの名前
        :param model_cls: モデルのクラス
        :param features: 特徴量のリスト
        :param params: ハイパーパラメータ
        :param x_train_path: 説明変数の学習データのパス
        :param y_train_path: 目的変数の学習データのパス
        :param x_test_path: テストデータのパス
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.features = features
        self.params = params
        self.x_train_path = x_train_path
        self.y_train_path = y_train_path
        self.x_test_path = x_test_path
        self.n_fold = 4

    def _run_train_fold(
        self, i_fold: Union[int, str]
    ) -> Tuple[Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """クロスバリデーションでのfoldを指定して学習・評価を行う

        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        validation = i_fold != "all"
        train_x = self._load_x_train()
        train_y = self._load_y_train()

        if validation:
            # 学習データ・バリデーションデータをセットする
            tr_idx, va_idx = self._load_index_fold(i_fold)
            tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
            va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

            # 学習を行う
            model = self._build_model(i_fold)
            model.train(tr_x, tr_y, va_x, va_y)

            # バリデーションデータへの予測・評価を行う
            va_pred = model.predict(va_x)
            score = log_loss(va_y, va_pred, eps=1e-15, normalize=True)

            # モデル、インデックス、予測値、評価を返す
            return model, va_idx, va_pred, score
        else:
            # 学習データ全てで学習を行う
            model = self._build_model(i_fold)
            model.train(train_x, train_y)

            # モデルを返す
            return model, None, None, None

    def run_train_cv(self) -> None:
        """クロスバリデーションでの学習・評価を行う

        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        logger.info(f"{self.run_name} - start training cv")

        scores = []
        va_idxes = []
        preds = []

        # 各foldで学習を行う
        for i_fold in range(self.n_fold):
            # 学習を行う
            logger.info(f"{self.run_name} fold {i_fold} - start training")
            model, va_idx, va_pred, score = self._run_train_fold(i_fold)
            logger.info(f"{self.run_name} fold {i_fold} - end training - score {score}")

            # モデルを保存する
            model.save_model()

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f"{self.run_name} - end training cv - score {np.mean(scores)}")

        # 予測結果の保存
        Util.dump(preds, f"../model/pred/{self.run_name}-train.pkl")

        # 評価結果の保存
        logger.result_scores(self.run_name, scores)

    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う

        あらかじめrun_train_cvを実行しておく必要がある
        """
        logger.info(f"{self.run_name} - start prediction cv")

        test_x = self._load_x_test()

        preds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_fold):
            logger.info(f"{self.run_name} - start prediction fold:{i_fold}")
            model = self._build_model(i_fold)
            model.load_model()
            pred = model.predict(test_x)
            preds.append(pred)
            logger.info(f"{self.run_name} - end prediction fold:{i_fold}")

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        Util.dump(pred_avg, f"../models/pred/{self.run_name}-test.pkl")

        logger.info(f"{self.run_name} - end prediction cv")

    def run_train_all(self) -> None:
        """学習データすべてで学習し、そのモデルを保存する"""
        logger.info(f"{self.run_name} - start training all")

        # 学習データ全てで学習を行う
        i_fold = "all"
        model, _, _, _ = self._run_train_fold(i_fold)
        model.save_model()

        logger.info(f"{self.run_name} - end training all")

    def run_predict_all(self) -> None:
        """学習データすべてで学習したモデルにより、テストデータの予測を行う

        あらかじめrun_train_allを実行しておく必要がある
        """
        logger.info(f"{self.run_name} - start prediction all")

        test_x = self._load_x_test()

        # 学習データ全てで学習したモデルで予測を行う
        i_fold = "all"
        model = self._build_model(i_fold)
        model.load_model()
        pred = model.predict(test_x)

        # 予測結果の保存
        Util.dump(pred, f"../models/pred/{self.run_name}-test.pkl")

        logger.info(f"{self.run_name} - end prediction all")

    def _build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う

        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f"{self.run_name}-{i_fold}"
        return self.model_cls(run_fold_name, self.params)

    def _load_x_train(self) -> pd.DataFrame:
        """学習データの特徴量を読み込む

        :return: 学習データの特徴量
        """
        # 学習データの読込を行う
        # 列名で抽出する以上のことを行う場合、このメソッドの修正が必要
        return pd.read_csv(self.x_train_path)[self.features]

    def _load_y_train(self) -> pd.Series:
        """学習データの目的変数を読み込む

        :return: 学習データの目的変数
        """
        # 目的変数の読込を行う
        train_y = pd.read_csv(self.y_train_path)["target"]
        # FIXME: ここが何で-1をしているかが分からない
        train_y = np.array([int(st[-1]) for st in train_y]) - 1
        train_y = pd.Series(train_y)
        return train_y

    def _load_x_test(self) -> pd.DataFrame:
        """テストデータの特徴量を読み込む

        :return: テストデータの特徴量
        """
        return pd.read_csv(self.x_test_path)[self.features]

    def _load_index_fold(self, i_fold: int) -> np.array:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す

        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        # ここでは乱数を固定して毎回作成しているが、ファイルに保存する方法もある
        train_y = self._load_y_train()
        dummy_x = np.zeros(len(train_y))
        skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=71)
        return list(skf.split(dummy_x, train_y))[i_fold]
