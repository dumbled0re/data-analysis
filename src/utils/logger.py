import datetime
import logging

import numpy as np


class Logger:
    def __init__(self):
        self.general_logger = logging.getLogger("general")
        self.result_logger = logging.getLogger("result")
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler("models/log/general.log")
        file_result_handler = logging.FileHandler("models/log/result.log")
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        # 時刻をつけてコンソールとログに出力
        self.general_logger.info("[{}] - {}".format(self.now_string(), message))

    def result(self, message):
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name, scores):
        # 計算結果をコンソールと計算結果用ログに出力
        dic = dict()
        dic["name"] = run_name
        dic["score"] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f"score{i+1}"] = score
        self.result(self.to_ltsv(dic))

    def now_string(self):
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_ltsv(self, dic):
        return " ".join([f"{key}: {value}" for key, value in dic.items()])
