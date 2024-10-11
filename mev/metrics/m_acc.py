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
from ad_metric_base import AnomalyDetectionMetricBase


class MetricACC(AnomalyDetectionMetricBase):
    def __init__(self, step_name: str, cfg: dict, logger: Log):
        super().__init__(step_name, cfg, logger)

    def run(self, data: pd.DataFrame, file_name: str, graph: bool, *args) -> float:
        """
        Method computes accuracy score based on the supplied data frame with
        ground truth and prediction values computed for each ROC threshold(point)
        that is result of anomaly detection process delivered by AD model.
        Args:
            data: DataFrame containing
            file_name: output file where metric should be saved or None if there is no need to save
                       anything
            graph: flag enables graphing given metric
        Returns:
            <object>
        """
        trace_gt = data.loc[:, self.lab_ground_truth].copy(False).tolist()
        trace_pr = data.loc[:, self.lab_prediction].copy(False).tolist()
        acc = metrics.accuracy_score(trace_gt, trace_pr)

        return acc
