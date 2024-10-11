from __future__ import annotations


import os
import sys

from sklearn import metrics
import pandas as pd
from log import Log

PATHS = ["../", "../tools", "../metrics"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from met_base import AnomalyDetectionMetricsBase

# from tools.error_reporter import ErrorReporter
from tools.plotter_helper import Plotter

from tools.roc_util import RocUtil
from tools.iter_helper import IterHelper


class AnomalyDetectionMetricsPerModel(AnomalyDetectionMetricsBase):
    """Class defining anomaly detection metrics for ADEF framework"""

    def __init__(self, step_name: str, cfg: dict, logger: Log) -> None:
        """
        CTOR for per Model metric class
        Args:
            step_name: name of the step
            cfg: main configuration object
            logger: logger object
        Returns:
            <None>
        """
        super().__init__(step_name, cfg, logger)
        self.limit = self.cfg["process"]["limit"]

    def run(self,
            results: pd.DataFrame,
            best_point: dict,
            training: bool,
            line_enabled: bool,
            m_params: dict = None,
            graphs: bool = True) -> dict:
        """
        Method runs all metrics required in given step to compute and generate results.
        Args:
            results: data frame containing combined model run.
            best_point: data frame containing best results achieved in the model.
            training: flag to determine whether record data
            line_enabled: flag to determine how plots will be drawn (plot - True, scatter - False)
            m_params: additional argument parameters stored in dictionary (key/value pairs)
                      accessible via args[0]
            graphs: Flag that enables graph generation or not
        Returns:
            <dict>: dictionary containing AD model metrics derived from the ROC curve/data
        """
        cfg = best_point["cfg"]
        result_file = best_point["file"]
        model = best_point["model"]
        fpr = results.loc[:, self.cfg["data_labels"]["fpr"]].copy(False).tolist()
        tpr = results.loc[:, self.cfg["data_labels"]["tpr"]].copy(False).tolist()

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

        # m_common = self.short_run("", results)
        try:
            auc_val = dict(self.get().items())["auc"].run(
                results, f"{result_file[:-4]}_auc.png", True, params
            )
        except Exception as ex:
            self.log.exception(ex)

        opt_box = RocUtil.opt_detection_box(results, self.cfg["process"]["limit"] / 100)

        model_metric = {
            "cfg": cfg,
            "model": model,
            "file": result_file,
            "auc": auc_val,
            "acc": round(best_point["acc"], 2),
            "f1": round(best_point["f1"], 2),
            "eer": round(best_point["eer"], 2),
            "prec": round(best_point["prec"], 2),
            "recall": round(best_point["recall"], 2),
            "tpr0": opt_box[0][1],
            "tprn": opt_box[1][1],
            "d1_min": best_point["d1_min"],
            "d1_avg": best_point["d1_avg"],
            "d1_max": best_point["d1_max"],
            "d2_min": 0 if "d2_min" not in best_point.keys() else best_point["d2_min"],
            "d2_avg": 0 if "d2_avg" not in best_point.keys() else best_point["d2_avg"],
            "d2_max": 0 if "d2_max" not in best_point.keys() else best_point["d2_max"],
        }

        fpr_label = self.cfg["data_labels"]["fpr"]
        tpr_label = self.cfg["data_labels"]["tpr"]

        label_str = model_metric["file"][model_metric["file"].rfind("/") + 1:].replace("_roc_cd", "")

        names = IterHelper.extract_names(label_str)
        algo_name = names["algo_name"]
        attack_name = names["attack_name"]
        model = 0 if not names["model_no"] else int(names["model_no"])
        parameter = 0 if not names["parameter_no"] else int(names["parameter_no"])

        if "step5" != self.step_name.lower():
            results["tau"] = 0

        # Fill model information and clear NaN values
        results.loc[:, ["model", "file"]] = results.loc[:, ["model", "file"]].fillna("")
        results.loc[:, ["cfg"]] = results.loc[:, ["cfg"]].fillna("{}")

        results.fillna(0.0, inplace=True)
        results.to_csv(f"{result_file[:-4]}_roc.csv", index=False)

        if graphs:
            prefix_name = f"Model: {algo_name.upper()} - Attack:{attack_name.upper()} -"

            Plotter.d1_over_tau(
                results.loc[:, ["fpr", "tpr", "d1_min", "d1_avg", "d1_max"]],
                f"{prefix_name} Δ1 M[{model}] P[{parameter}]",
                f"{result_file[:-4]}_d1.png",
                line_enabled
            )

            # Per model graph
            Plotter.roc_2d(
                results.loc[:, [fpr_label, tpr_label]],
                model_metric,
                f"{prefix_name} ROC M[{model}] P[{parameter}]",
                f"{result_file[:-4]}_roc.png",
                self.limit / 100,
                line_enabled
            )

            if self.cfg["model"]["animation"]:
                Plotter.roc_2d_anim(
                    results.loc[:, [fpr_label, tpr_label]],
                    model_metric,
                    f"{prefix_name} ROC M[{model}] P[{parameter}]",
                    f"{result_file[:-4]}_roc_anim.gif",
                    self.limit / 100,
                    line_enabled
                )

            Plotter.cf_insight(
                results.loc[results.index[1] : results.index[-1], :],
                f"{prefix_name} Confusion Matrix over threshold",
                f"{result_file[:-4]}_cf.png",
            )

            Plotter.precision_recall(
                results.loc[:, ["prec", "recall"]],
                f"{prefix_name} Precision/Recall",
                f"{result_file[:-4]}_prec_recall.png",
                line_enabled
            )

        return model_metric
