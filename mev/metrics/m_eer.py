
from __future__ import annotations

import os
import sys
import pandas as pd
import math
from scipy.optimize import brentq
from scipy.interpolate import interp1d

PATHS = ["../", "../tools", "../metrics"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from log import Log
from ad_metric_base import AnomalyDetectionMetricBase


class MetricEER(AnomalyDetectionMetricBase):
    def __init__(self, step_name: str, cfg: dict, logger: Log):
        super().__init__(step_name, cfg, logger)

    def run(
        self,
        data: pd.DataFrame,
        file_name: str,
        graph: bool,
        params: dict,
        *args,
    ) -> float:
        """
        Method computes AUC value based on the supplied data frame with
        TPR/FPR values computed for each ROC threshold that is result of
        anomaly detection process delivered by AD model.
        Args:
            data: DataFrame containing
            file_name: output file where metric should be saved or None if there is no need to save
                       anything
            graph:
            params:
            args:
        Returns:
            <object>
        """
        eer = 0.5

        if not any(math.isnan(x) for x in params["fpr"]) and not any(
            math.isnan(x) for x in params["tpr"]
        ):
            try:
                eer = brentq(
                    lambda x: 1.0 - x - interp1d(params["fpr"], params["tpr"])(x),
                    0.0,
                    1.0,
                )
            except Exception as ex:
                print(ex)
        return eer
