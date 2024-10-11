from __future__ import annotations


import os
import sys
import shutil
import glob

import pandas as pd
import numpy as np

from log import Log

PATHS = ["../", "../tools", "../metrics"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from plotter_helper import Plotter
from reg_service import RegService


class AnomalyDetectionMetricsDiff:
    """
    Class defining anomaly detection differential
    metrics for ADEF framework.
    """

    def __init__(self, step_name: str, cfg: dict, logger: Log) -> None:
        """
        CTOR
        Args:
            step_name: name of the simulation step
            cfg: configuration object
            logger: logger object
        Returns:
            <None>
        """
        self.log = logger
        self.step_name = step_name
        self.cfg = cfg
        self.path = f"{self.cfg['global']['path']}experiments/{self.cfg['experiment']['path']}"
        # algo_name_map = {x: "".join(x.split("_")[1:]) if "_" in x else x for x in algo_names}

        self.algo_names_map = {k.upper(): v["short_name"].upper() for k, v in self.cfg["model"]["ad"].items()
                               if v["enabled"]}

    def record(
        self,
        prob_set_str: str,
        algo_name: str,
        attack_name: str,
        metric_type: str,
        knowledge: bool,
        metric_data: pd.DataFrame,
        metric_record: dict,
        out_file: str
    ) -> pd.DataFrame:
        """
        Method records computed metrics that will be later used
        to build results and comparison data.
        Args:
            prob_set_str: str
            algo_name: str
            attack_name: str
            metric_type: str
            knowledge: bool
            metric_data: pd.DataFrame
            metric_record: dict
            out_file: str
        Returns:
            <pd.DataFrame> - updated metric_data
        """

        # Best parameters are used to detect an anomaly and generate data for the
        # delta 1 and delta 2 computation

        metric_vals = {
            "prob": [prob_set_str],
            "algo": [algo_name],
            "attack": [attack_name],
            "type": [metric_type],
            "knowledge": [str(knowledge)],
        }
        if not metric_record:
            self.log.error(
                f"Metric is empty: |{metric_record}|. Check the detection algorithm"
            )
            return metric_data

        metric_vals.update(
            {
                key: [round(value, 2)] if isinstance(value, float) else [value]
                for key, value in metric_record.items()
            }
        )

        metric_data = (
            pd.concat([metric_data, pd.DataFrame(metric_vals)])
            if metric_data.shape[0] > 0
            else pd.DataFrame(metric_vals)
        )

        if metric_data.isna().all().all():
            metric_data.replace(np.nan, 0.0, inplace=True)

        filter_metric_data = metric_data[metric_data["prob"] == prob_set_str]

        if os.path.isfile(out_file):
            file_metric_data = pd.read_csv(out_file)
            mask = (filter_metric_data[["prob", "algo", "attack", "type", "knowledge"]]
                    .isin({"prob": metric_vals["prob"],
                           "algo": metric_vals["algo"],
                           "attack": metric_vals["attack"],
                           "type": metric_vals["type"],
                           "knowledge": metric_vals["knowledge"]})
                    .all(axis=1))

            if not mask.any():
                filter_metric_data = pd.concat([file_metric_data, filter_metric_data])
                for col in filter_metric_data.columns:
                    filter_metric_data[col] = filter_metric_data[col].apply(lambda d: str(d) if isinstance(d, dict) else d)

        filter_metric_data.to_csv(out_file, index=False)

        return metric_data

    def run(self) -> None:
        """
        Method generates difference graphs for series of metrics computed between steps.
        Args:
            <None>
        Returns:
            <None>
        """
        for prob_set in self.cfg["attack"]["densities"]:
            probability = f"{str(int(prob_set * 100))}"
            metric_files = glob.glob(
                f"{self.path}{self.cfg['results']['path']}a_data_{probability}_metric.csv"
            )

            # --------------------- Sum data -----------------------
            sum_data = None
            prob_set_str = None

            for metric_file in metric_files:
                prob_set_str = metric_file[metric_file.rfind("/") :].split("_")[2]
                attack_count = len(
                    {
                        k: v
                        for k, v in self.cfg["attack"]["types"].items()
                        if 1 == v["enabled"]
                    }.keys()
                )
                sum_data_row = pd.read_csv(metric_file)

                # Inserting amount of attacks used in training set
                sum_data_row["attack_no"] = attack_count

                if sum_data is None:
                    sum_data = sum_data_row
                else:
                    sum_data = pd.concat([sum_data, sum_data_row])

                self.log.debug(f"Processed file: {metric_file}")

            cycle_no = self.cfg["process"]["cycles"]

            if prob_set_str is None:
                return

            sum_file_name = (
                f"{self.path}{self.cfg['results']['path']}"
                f"final_{cycle_no}_{prob_set_str}_sum"
            )
            sum_data.fillna(0.0, inplace=True)
            # TODO: Validate sum data, and in case if this doesn't work do not save the data set + warning
            validation = True

            if not validation:
                self.log.debug(
                    f"File {sum_file_name}.csv, will not be saved - validation FAILED. Data shape: {sum_data.shape}"
                )
                return
            # sum_data.drop_duplicates(inplace=True)
            sum_data.to_csv(f"{sum_file_name}.csv", index=False)

            self.log.debug(f"Using sum_file: {sum_file_name}.csv")

            # Generate final image for algorithms comparison
            for label in ["auc", "acc", "f1", "eer"]:
                Plotter.sum_heatmap(self.algo_names_map,
                                    sum_data,
                                    f"{label.upper()} comparison",
                                    f"{sum_file_name}_{label}.png",
                                    self.log,
                                    label)

                Plotter.gain_loss_heatmap(self.algo_names_map,
                                          sum_data,
                                          f"{label.upper()} Gain/loss comparison",
                                          f"{sum_file_name}_{label}_gain_loss.png",
                                          self.log,
                                          label,
                                          True if "eer" == label else False)
