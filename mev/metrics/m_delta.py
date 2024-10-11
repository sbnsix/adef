from __future__ import annotations


import os
import sys
import pandas as pd

from numpy.lib.function_base import average

PATHS = [
    "./",
    "../",
    "../../tools",
    "../data",
]

for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from log import Log
from ad_metric_base import AnomalyDetectionMetricBase
from eng import DataEng


class MetricDelta:
    """
    Class wrapper used for delta metrics computations
    """

    @staticmethod
    def load_cycles(data: pd.DataFrame, cycle_length: int) -> list:
        """
        Method loads cycles from given CSV file
        Args:
            data:           input data frame that will be split into cycles
            cycle_length:   detection cycle length
        Returns:
            <list> - Original DataFrame and list of pd.DataFrame chunks divided
                     in accordance to cycle configuration
        """
        # Add cycle detector component evaluation result column
        if "cd_result" not in data.columns:
            data["cd_result"] = 0

        # Divide data set into cycles
        data_list = DataEng.split_data(data, cycle_length)

        return data_list

    @staticmethod
    def check_continuity(s_indices: list):
        """
        Args:
            s_indices: list of indices to be validated
        Returns:
            <list> - list of lists of indices containing continuous indexes
        """
        prev_i = 0
        cnt_idx = 0
        first_idx = 0
        slices = []

        for i in s_indices:
            if cnt_idx == 0:
                cnt_idx += 1
                prev_i = i
                continue

            if i - prev_i > 1:
                slc = s_indices[first_idx:cnt_idx]
                if len(slc) > 0:
                    slices.append(slc)
                first_idx = cnt_idx + 1

            prev_i = i
            cnt_idx += 1

        slc = s_indices[first_idx : len(s_indices) - 1]
        if len(slc) > 0:
            slices.append(slc)

        return slices

    @staticmethod
    def compute_delta(
        data_list: list(pd.DataFrame), grt_column: str, pre_column: str, tau: int
    ) -> list:
        """
        Method computes delta one across all traces
        Args:
            data_list  : list of data frames containing autoclave profiles
            grt_column : ground truth column used to count difference from
            pre_column : prediction column used to compute delta
            tau        : range of point considered during delta computation
        Returns:
            <list> - list of precomputed deltas between ground truth
                     and detections performed by AD models
        """
        deltas = []
        if data_list is None:
            return [0, 0, 0]

        for data_c in data_list:
            d_diff = data_c[
                (data_c.loc[:, grt_column] == 1)
                & (data_c.loc[:, pre_column] == 0)
                & (data_c.loc[:, grt_column].sum() > 0)
                & (data_c.loc[:, pre_column].sum() > 0)
            ]

            if d_diff.shape[0] == 0:
                continue

            # Graph shaping to select raising slope with tau value
            s_indices = data_c.loc[(data_c.loc[:, grt_column] == 1)].index

            # Mechanics to find indexes that are not continuous and process them
            c_indexes = MetricDelta.check_continuity(s_indices)

            # Iterate over all of, the list of indexes
            for c_index in c_indexes:
                start_idx = (
                    c_index[0] - tau
                    if data_c.index[0] > c_index[0] - tau
                    else data_c.index[0]
                )
                stop_idx = (
                    c_index[0] + tau
                    if data_c.index[-1] > c_index[0] + tau
                    else data_c.index[-1]
                )

                data_idx = data_c.loc[start_idx:stop_idx,]

                d = 0
                sign = 1
                for idx, r in data_idx.iterrows():
                    # Sign evaluation
                    if r[grt_column] == 0 and r[pre_column] == 1:
                        sign = -1
                    # elif r[grt_column] == 0 and r[pre_column] == 1:
                    #    sign = 1

                    if r[grt_column] == r[pre_column] == 0:
                        d = 0
                        sign = 1
                    elif r[grt_column] == 1 and r[pre_column] == 1:
                        dx = d * sign
                        if d > 0 and dx > 0:
                            deltas.append(dx)
                            d = 0
                            sign = 1
                    else:
                        d += 1

        if len(deltas) == 0:
            return [0, 0, 0]

        comp_deltas = [min(deltas), round(average(deltas), 2), max(deltas)]

        return comp_deltas


class MetricD1(AnomalyDetectionMetricBase):
    def __init__(self, step_name: str, cfg: dict, logger: Log):
        super().__init__(step_name, cfg, logger)
        self.cycle_len = (
            self.cfg["process"]["cycle_len"] * self.cfg["process"]["samples"]
        )

    def run(self, data: pd.DataFrame, file_name: str, graph: bool, *args) -> dict:
        """
        Method computes AUC value based on the supplied data frame with
        TPR/FPR values computed for each ROC threshold that is result of
        anomaly detection process delivered by AD model.
        Args:
            data: DataFrame containing AD model data for given trace
            file_name: output file where metric should be saved or None if there is no need to save
                       anything
            graph:
            args:
        Returns:
            <pd.DataFrame> - DataFrame object containing information about
                             value, min, max and average setting for D1 metric
        """
        max_search_limit = (
            int(args[0]["max_search_limit"])
            if "max_search_limit" in args[0].keys()
            else 30
        )

        # Compute Delta 1 (comparison between GT and AD detection)
        data_list = MetricDelta.load_cycles(data, self.cycle_len)

        d1s = MetricDelta.compute_delta(
            data_list, self.lab_ground_truth, self.lab_prediction, max_search_limit
        )
        d1s = [round(d, 2) for d in d1s]

        return {"d1_min": d1s[0], "d1_avg": d1s[1], "d1_max": d1s[2]}


class MetricD2(AnomalyDetectionMetricBase):
    def __init__(self, step_name: str, cfg: dict, logger: Log):
        super().__init__(step_name, cfg, logger)
        self.cycle_len = (
            self.cfg["process"]["cycle_len"] * self.cfg["process"]["samples"]
        )

    def run(self, data: pd.DataFrame, file_name: str, graph: bool, *args) -> dict:
        """
        Method computes AUC value based on the supplied data frame with
        TPR/FPR values computed for each ROC threshold that is result of
        anomaly detection process delivered by AD model.
        Args:
            data: data containing input to compute the metric - not used in this case
            file_name: output file where metric should be saved or None if there is no need to save
                       anything
            graph: boolean flag to graph the metric
            args: other metric configuration parameters
        Returns:
            <dict> - object containing information about
                     tau value, min, max and average setting for D2 metric
        """
        max_search_limit = (
            int(args[0]["max_search_limit"])
            if "max_search_limit" in args[0].keys()
            else 30
        )
        tau = int(args[0]["tau"]) if "tau" in args[0].keys() else 0

        # The search limit is adjusted inside delta computation.
        max_search_limit += tau

        d2s = MetricDelta.compute_delta(
            args[0]["data_list"], "label", "cd_result", max_search_limit
        )
        d2s = [round(d, 2) for d in d2s]

        return {"tau": tau, "d2_min": d2s[0], "d2_avg": d2s[1], "d2_max": d2s[2]}
