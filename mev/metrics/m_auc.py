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


class MetricAUC(AnomalyDetectionMetricBase):
    def __init__(self, step_name: str, cfg: dict, logger: Log):
        super().__init__(step_name, cfg, logger)

    def run(self,
            data: pd.DataFrame,
            file_name: str,
            graph: bool,
            params: dict,
            *args) -> float:
        """
        Method computes AUC value based on the supplied data frame with
        TPR/FPR values computed for each ROC threshold that is result of
        anomaly detection process delivered by AD model.
        Args:
            data: DataFrame containing
            file_name: output file where metric should be saved or None if there is no need to save
                       anything
            graph: flag enables graphing given metric
            params:
            args:
        Returns:
            <object>
        """
        auc = metrics.auc(params["fpr"], params["tpr"])
        return auc
