from __future__ import annotations

import pandas as pd
from log import Log


class AnomalyDetectionMetricBase:
    """Class defining anomaly detection metric base"""

    def __init__(self, step_name: str, cfg: dict, logger: Log) -> None:
        """
        CTOR
        Args:
            cfg: configuration object
            logger: logger object
        Returns:
            <None>
        """
        self.cfg = cfg
        self.log = logger
        self.step_name = step_name

        self.lab_ground_truth = self.cfg["data_labels"][
            "ad" if "step5" != step_name.lower() else "cd"
        ]["ground_truth"]
        self.lab_prediction = self.cfg["data_labels"][
            "ad" if "step5" != step_name.lower() else "cd"
        ]["prediction"]

    def run(self, data: pd.DataFrame, file_name: str, graph: bool, *args) -> object:
        """
        Method computes given metric based on the given data
        Args:
            data: DataFrame containing
            file_name: output file where metric should be saved or None if there is no need to save
                       anything
            graph: flag enables graphing given metric
            args: other customised parameters used in the metric computation
        Returns:
            <object>
        """
        raise NotImplemented(
            "Please add run method to specific implementation of -> AnomalyDetectionMetricBase.run()"
        )
