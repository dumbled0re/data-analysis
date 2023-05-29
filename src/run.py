from model.model_xgb import ModelXGB
from runner import Runner
from utils import config
from utils.submission import Submission

if __name__ == "__main__":
    # xgboostによる学習・予測
    runner = Runner(
        run_name="xgb1",
        model_cls=ModelXGB,
        features=config.features,
        params=config.rg_params_xgb,
        task_type="regression",
        x_train_path="data/processed/train_culture_medium.csv",
        y_train_path="data/processed/train_culture_medium.csv",
        x_test_path="data/processed/test_culture_medium.csv",
    )
    runner.run_train_cv()
    runner.run_predict_cv()
    submission = Submission(
        run_name="xgb1",
        submission_file_path="data/processed/test_culture_medium_for_submission.csv",
    )
    submission.create_submission()

    # ニューラルネットによる学習・予測
    # runner = Runner("nn1", ModelNN, features, params_nn)
    # runner.run_train_cv()
    # runner.run_predict_cv()
    # Submission.create_submission("nn1")

    """
    # (参考）xgboostによる学習・予測 - 学習データ全体を使う場合
    runner = Runner('xgb1-train-all', ModelXGB, features, params_xgb_all)
    runner.run_train_all()
    runner.run_test_all()
    Submission.create_submission('xgb1-train-all')
    """
