import pandas as pd

from .util import Util


class Submission:
    @classmethod
    def create_submission(cls, run_name):
        submission = pd.read_csv("../../data/raw/sampleSubmission.csv")
        pred = Util.load(f"../models/pred/{run_name}-test.pkl")
        for i in range(pred.shape[1]):
            submission[f"Class_{i + 1}"] = pred[:, i]
        submission.to_csv(f"../submission/{run_name}.csv", index=False)
