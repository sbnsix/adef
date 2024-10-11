""" Threat tools library used in attack modelling """

from __future__ import annotations
from typing import Tuple
import pandas as pd
import numpy as np
from threat_class import ThreatClass


class ThreatTools:
    """
    Threat Modelling Tool set class containing mathematical function to model attacks
    """

    @staticmethod
    def res_exp(x: list, y: list, pw: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resolve exponent parameters between two points on 2D space
        Args:
            x - start/stop X dimension
            y - start/stop Y dimension
            pw - exp power
        Returns:
            (a, b) - tuple for exp function parameters that fits two points
        """
        A = np.exp(np.log(y[0] / y[1]) / pw)
        a = (x[0] - x[1] * A) / (A - 1)
        b = y[0] / (x[0] + a) ** pw

        return (a, b)

    @staticmethod
    def exp_func(x: list,
                 a: float,
                 b: float,
                 power: int) -> list:
        """
        Exponential function
        Args:
            x: X axis
            a: exp coefficient
            b: exp coefficient
            power: power used in exp computation
        Returns:
            <list> - values on Y axis
        """
        return list(((x + a) ** power) * b)

    @staticmethod
    def soft_filter(attack_name: str,
                    data: pd.DataFrame,
                    sf_cfg: dict) -> pd.DataFrame:
        """
        Method softens sharp edges of flat attack to better reflect
        physics based autoclave profile behavior
        Args:
            attack_name     - name of the attack
            data            - data with flat attack that needs to be softened
            sf_cfg          - soft filter configuration
        Returns:
            <pd.DataFrame>  - soften trace that will be more likely to reflect
                              physics based modelling
        """
        # Soften trace as flat trace is not something that should stick
        # to physics-based modelling
        # Data 0 and data N are main approximation points
        label = "tc"
        s_data = data.copy(deep=True)

        power_val = int(sf_cfg["power"])
        # Remove start and end as they are original trace
        d_len = len(s_data.loc[:]) - 1
        soft_range = int(d_len / 5)

        # Determine which part of the trace is required to be "soften"
        # Smooth start of the attack
        distance = abs(
            s_data.loc[s_data.index[0], label]
            - s_data.loc[s_data.index[soft_range], label]
        )
        if distance > sf_cfg["limit"]:
            # self.log.debug(f"C1 distance {distance:.2f}")

            a, b = ThreatTools.res_exp(
                [s_data.index[0], s_data.index[soft_range]],
                [
                    s_data.loc[s_data.index[0], label],
                    s_data.loc[s_data.index[soft_range], label],
                ],
                power_val,
            )

            s_data.loc[
                s_data.index[1] : s_data.index[soft_range], label
            ] = ThreatTools.exp_func(
                range(s_data.index[0], s_data.index[soft_range]), a, b, power_val
            )

        # Skip smoothing the end if fan is turned off
        if attack_name == "doff":
            return s_data

        # Smooth end of the attack
        distance = abs(
            s_data.loc[s_data.index[d_len - soft_range], label]
            - s_data.loc[s_data.index[d_len], label]
        )
        if distance > sf_cfg["limit"]:
            a, b = ThreatTools.res_exp(
                [s_data.index[d_len - soft_range - 1], s_data.index[d_len]],
                [
                    s_data.loc[s_data.index[d_len - soft_range - 1], label],
                    s_data.loc[s_data.index[d_len], label],
                ],
                power_val,
            )
            s_data.loc[
                s_data.index[d_len - soft_range] : s_data.index[d_len], label
            ] = ThreatTools.exp_func(
                range(s_data.index[d_len - soft_range - 1], s_data.index[d_len]),
                a,
                b,
                power_val,
            )

        return s_data

    @staticmethod
    def mark_ground_truth(data_input: pd.DataFrame,
                          data_output: pd.DataFrame,
                          column: str,
                          cfg: dict,
                          a_start: int,
                          a_stop: int,
                          th_class: ThreatClass) -> None:
        """
        Method computing
        Args:
            data_input: original signal without any disturbances
            data_output: malformed signal - after attack changes
            column: column name where data set can be found
            cfg: attack configuration object containing ground truth evaluation rules
            a_start: attack start
            a_stop: attack stop
        Output:
            <pd.Series>: vector containing ground truth labels or None
        """
        # Mark type of the threat
        data_output.loc[a_start: a_stop, "type"] = th_class

        if cfg["ground_truth"] == "default":
            # Mark attack area
            data_output.loc[a_start: a_stop, "label"] = 1
        elif cfg["ground_truth"] == "auto":
            percentage = (cfg["gt_std_dev"]/100)
            threshold = (data_output.loc[:, column]-data_input.loc[:, column]).abs().std()*percentage
            data_output.loc[:, "label"] = np.where((data_output.loc[:, column]-data_input.loc[:, column]).abs() >
                                                    threshold, 1, 0).astype(int)
        elif cfg["ground_truth"] == "range":
            percentage = (cfg["gt_std_dev"] / 100)
            threshold = (data_output.loc[a_start:a_stop, column] -
                         data_input.loc[a_start:a_stop, column]).abs().std()*percentage
            data_output.loc[a_start:a_stop, "label"] = np.where((data_output.loc[a_start:a_stop, column] -
                                                                 data_input.loc[a_start:a_stop, column]).abs() >
                                                                 threshold, 1, 0).astype(int)
