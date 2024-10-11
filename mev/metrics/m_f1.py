from __future__ import annotations


import os
import sys
import pandas as pd
from sklearn import metrics

PATHS = ["../", "../tools", "../metrics"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from log import Log
from plotter_helper import Plotter
from ad_metric_base import AnomalyDetectionMetricBase


class MetricF1(AnomalyDetectionMetricBase):
    def __init__(self, step_name: str, cfg: dict, logger: Log):
        super().__init__(step_name, cfg, logger)

    def run(self, data: pd.DataFrame, file_name: str, graph: bool, *args) -> float:
        """
        Method computes AUC value based on the supplied data frame with
        TPR/FPR values computed for each ROC threshold that is result of
        anomaly detection process delivered by AD model.
        Args:
            data: DataFrame containing
            file_name: output file where metric should be saved or None if there is no need to save
                       anything
        Returns:
            <object>
        """
        trace_gt = data.loc[:, self.lab_ground_truth].copy(False).tolist()
        trace_pr = data.loc[:, self.lab_prediction].copy(False).tolist()

        f1_score = metrics.f1_score(trace_gt, trace_pr, zero_division=1)

        return f1_score
