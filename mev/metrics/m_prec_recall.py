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


class MetricPrecRecall(AnomalyDetectionMetricBase):
    def __init__(self, step_name: str, cfg: dict, logger: Log):
        super().__init__(step_name, cfg, logger)

    def run(
        self, data: pd.DataFrame, file_name: str, graph: bool, params: dict, *args
    ) -> dict:
        """
        Method computes AUC value based on the supplied data frame with
        TPR/FPR values computed for each ROC threshold that is result of
        anomaly detection process delivered by AD model.
        Args:
            data: DataFrame containing
            file_name: output file where metric should be saved or None if there is no need to save
                       anything
            graph: bool
            params: dict, *args
        Returns:
            <dict> - recall/precision parameters
        """
        gnd_truth = data.loc[:, params["gnt"]]
        pred_result = data.loc[:, params["pred"]]
        (
            prec_val,
            recall_val,
            fbeta_score,
            support,
        ) = metrics.precision_recall_fscore_support(
            gnd_truth,
            pred_result,
            beta=1.0,
            labels=[0, 1],
            pos_label=1,
            average=None,
            zero_division=1,
        )

        prec_recall = {
            "prec": prec_val[1],
            "recall": recall_val[1],
            "fbeta": fbeta_score[1],
            "support": support[1],
        }

        return prec_recall
