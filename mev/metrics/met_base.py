from __future__ import annotations

import os
import sys
from sklearn import metrics

import pandas as pd

PATHS = ["../", "../tools", "../metrics"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from m_auc import MetricAUC
from m_acc import MetricACC
from m_eer import MetricEER
from m_f1 import MetricF1
from m_prec_recall import MetricPrecRecall
from m_delta import MetricD1, MetricD2
from tools.roc_util import RocUtil
from tools.log import Log
from tools.reg_service import RegService

class AnomalyDetectionMetricsBase(RegService):
    """Base class defining anomaly detection metrics for ADEF framework"""

    def __init__(self, step_name: str, cfg: dict, logger: Log) -> None:
        super().__init__(logger)
        self.metrics = {}
        self.cfg = cfg
        self.step_name = step_name
        self.lab_ground_truth = self.cfg["data_labels"][
            "ad" if "step5" != step_name.lower() else "cd"
        ]["ground_truth"]
        self.lab_prediction = self.cfg["data_labels"][
            "ad" if "step5" != step_name.lower() else "cd"
        ]["prediction"]
        self.cache = {}

        metric_mapper = {
            "auc": MetricAUC,
            "acc": MetricACC,
            "eer": MetricEER,
            "f1": MetricF1,
            "prec_recall": MetricPrecRecall,
            "d1": MetricD1,
            "d2": MetricD2,
        }

        # Register all available metrics from model point of view
        for metric_name in metric_mapper.keys():
            self.add(
                metric_name, metric_mapper[metric_name](self.step_name, cfg, self.log)
            )

    def metric_common(self,
                      file_name: str,
                      data: pd.DataFrame,
                      cfg: dict) -> (dict, list, list):
        """
        Common metric computed ROC
        Args:
            data: input data after completion of anomaly detection process,
            file_name: file name template for model and resulting data.
            cfg: given anomaly detection model parameter configuration.
        Returns:
            <dict> - metric data
            <list> - FPR - False Positive Rate
            <list> - TPR - True Positive Rate
        """
        data_metric = {"tn": 0, "fp": 0, "fn": 0, "tp": 0, "fpr": 0, "tpr": 0}
        # Evaluate algorithm based on result data
        # TPR, FPR
        trace_gt = data.loc[:, self.lab_ground_truth].copy(False).tolist()
        trace_pr = data.loc[:, self.lab_prediction].copy(False).tolist()

        # Compute single TPR/FPR point
        tn, fp, fn, tp = RocUtil.confusion_matrix(trace_gt, trace_pr)

        data_metric.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})

        try:
            fpr, tpr, ths = metrics.roc_curve(trace_gt, trace_pr)
        except Exception as ex:
            self.log.exception(ex)

        if len(fpr) < 2 and len(tpr) < 2:
            return data_metric, None, None

        data_metric.update({"fpr": fpr[1], "tpr": tpr[1]})

        data_metric["cfg"] = cfg
        data_metric["model"] = f"{file_name[:-4]}.pkl".replace("\\/", "/")
        data_metric["file"] = file_name.replace("\\/", "/")

        return data_metric, fpr, tpr

    def run(self,
            result_file: str,
            result: pd.DataFrame,
            cfg: dict,
            m_params: dict = None
    ) -> dict:
        """
        Method runs all metrics required in given step to compute and generate results.
        Args:
            result_file: data frame containing combined model run.
            result: result data frame containing current anomaly detection results from which
                    metrics will be computed.
            cfg: configuration object
            m_params: additional params that will be added to compute specific metrics.
        Returns:
            <dict>: dictionary containing AD metrics derived from the ROC curve
        """
        data_metric, fpr, tpr = self.metric_common(result_file, result, cfg)

        if fpr is None and tpr is None:
            return data_metric

        params = {
            "fpr": fpr,
            "tpr": tpr,
            "gnt": self.lab_ground_truth,
            "pred": self.lab_prediction,
            "max_search_limit": self.cfg["metrics"]["d1"]["max_search_limit"],
            "result_file": result_file,
        }

        if m_params is not None:
            params.update(m_params)

        # Apply filter to iterated metrics
        m_funcs = {
            k: v
            for (k, v) in self.get().items()
            if k not in self.cfg["metrics"]["filters"][self.step_name.lower()]
        }

        # Call each registered metric
        for name, metric in m_funcs.items():
            # Compute metric
            metric_value = metric.run(
                result,
                f"{result_file[:-4]}_{name}.png",
                False,
                params,
            )
            if isinstance(metric_value, dict):
                data_metric.update(metric_value)
            else:
                data_metric[name] = metric_value

        return data_metric
