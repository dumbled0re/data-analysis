import pandas as pd

from .util import Util


class Submission:
    def __init__(self, run_name: str, submission_file_path: str) -> None:
        self.run_name = run_name
        self.submission_file_path = submission_file_path

    def create_submission(self) -> None:
        submission = pd.read_csv(
            self.submission_file_path, engine="python", encoding="utf-8"
        )
        pred = Util.load(f"models/pred/{self.run_name}-test.pkl")
        # for i in range(pred.shape[1]):
        #     submission[f"Class_{i + 1}"] = pred[:, i]
        submission["pred"] = pred
        submission.to_csv(f"submission/{self.run_name}.csv", index=False)
