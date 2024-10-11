""" ADEF pre-simulation steps required to successfully experiment """

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

PATHS = ["./detectors", "./tools"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from tools.log import Log
import tools.yaml_converter as yc
from tools.mt import interpolate_slope


class PreProcessor:
    """
    Class defining input data preprocessing activities
    """

    def __init__(self, cfg: dict, logger: Log) -> PreProcessor:
        """
        CTOR
        Args:
            cfg             - PreProcessor configuration
            logger          - logger object
        Returns:
            <PreProcessor>  - instance of the clas
        """
        self.log = logger
        self.cfg = cfg

    def dissect(self) -> None:
        """
        Method used to validate various aspects of simulation engine
        Args:
            <None>
        Returns:
            <None>
        """
        break_cycle = self.cfg["process"]["cycle"]

        for file_path in self.cfg["file_paths"]:
            data = pd.read_csv(file_path[0])
            data.reset_index(inplace=True)
            # self.log.debug(f"Data shape: {str(data.shape)}")
            new_set = pd.DataFrame(columns=data.columns)

            i = 0

            # self.log.debug(f"Ranges: {str(cfg["ranges"])}")

            # Data selection and hike
            if self.cfg["ranges"] is not None:
                prev_range = None
                for range_set in self.cfg["ranges"]:
                    selection = pd.DataFrame(data.loc[range_set[0] : range_set[1], :])

                    adj = 0
                    if self.cfg["hike"] is not None:
                        if i < len(self.cfg["hike"]):
                            adj = self.cfg["hike"][i]

                    # Adjust only lower portion of the autoclave profile
                    # self.log.debug(f"Data: {str(selection.loc[(selection["tc"]+adj) < cfg["max_temp"], ["tc","temp"]])}")

                    selection.loc[
                        (selection["tc"] + adj) < self.cfg["process"]["max_temp"],
                        ["tc", "temp"],
                    ] = (
                        selection.loc[
                            (selection["tc"] + adj) < self.cfg["max_temp"],
                            ["tc", "temp"],
                        ]
                        + adj
                    )

                    # TODO: add break with linear approximation for next
                    # section start
                    break_section = None

                    if i > 0:
                        # Linear transition between sets to enable
                        # real model
                        tx = [
                            prev_range.loc[prev_range.index[-1], "time"],
                            prev_range.loc[prev_range.index[-1], "time"] + break_cycle,
                        ]
                        y = [
                            prev_range.loc[prev_range.index[-1], "tc"],
                            selection.loc[selection.index[0], "tc"],
                        ]

                        t1, y1 = interpolate_slope(tx, y)
                        # self.log.debug(f"{str(prev_range.loc[prev_range.index[-1], "tc"])} => {str(selection.loc[selection.index[0], "tc"])}")
                        # self.log.debug(f"Selection |{str(selection.loc[:, "tc"])}|")

                        # self.log.debug(f"t1: {str(t1)}")
                        # self.log.debug(f"t1 type: {str(type(t1))}")
                        # self.log.debug(f"t1 shape: {str(t1.shape)}")

                        # self.log.debug(f"y1 type: {str(type(y1))}")
                        # self.log.debug(f"y1 shape: {str(y1.shape)}")
                        # y1 = prev_range.loc[prev_range.index[-1], "tc"]
                        # self.log.debug(f"y1: {str(y1)}")

                        break_section = pd.DataFrame({"time": t1, "tc": y1, "temp": y1})
                        # break_section.to_csv(f"{file_path[1][:-4]}_{str(i)}.csv")

                    if break_section is not None:
                        new_set = pd.concat([new_set, break_section])

                    new_set = pd.concat([new_set, selection])
                    prev_range = selection
                    i += 1

            if len(new_set) == 0:
                new_set = data

            # Adjusting sampling rates
            # Use linear approximation to generate new data points
            if self.cfg["samples"] > 6:
                pass
            # Cut down data points from the set
            elif self.cfg["samples"] < 6:
                # Selects every 3rd raw starting from 0
                rate = 7 - self.cfg["samples"]
                if rate > 1:
                    new_set = pd.DataFrame(new_set[new_set.index % (rate) == 0])
                    # self.log.debug(f"New set shape: {str(new_set.shape)}")

            # Do not change anything
            else:
                pass
                # self.log.debug("Sampling rates adjusted")

            del new_set["index"]

            # Aligning all values to maximum temperature
            new_set.loc[
                (new_set["tc"] > self.cfg["max_temp"])
                | (new_set["temp"] > self.cfg["max_temp"]),
                ["tc", "temp"],
            ] = self.cfg["max_temp"]

            # Re-index whole data set to continuous time
            # so in simulation there will be no problems
            # related to infeasibility
            new_set["time"] = new_set["time"].round(decimals=0)
            delta = int(new_set["time"].diff().iloc[1])
            delta = delta if delta >= 1 else 1
            new_set["time"] = list(range(0, len(new_set), delta))
            new_set.set_index("time", inplace=True)

            self.log.debug(f"Writing input file: {file_path[1]}")
            new_set.to_csv(file_path[1])

        return

    def nth_linear_lbound(
        self, cfg: dict, data: pd.DataFrame, file_name: str = None
    ) -> pd.Series:
        """
        Method creates linear lower temperature bound for given trace
        Args:
            cfg:        list containing points of shape to be created on Y axis
            data:       input data that will be used to get X axis
            file_name:  name of the file
        Returns:
            <pd.Series> - new shape generated from linear approximation between
                          points generated
        """
        # 1. Create lower part shape over the time span
        # Using mathematical formulas
        # First approximation will be linear approximation
        # In the future a non-linear model shapes can be used
        # as dependency injection to create a trace

        nel = len(cfg) - 1
        nth_length = int(data.shape[0] / nel)
        # self.log.debug(f"N el: {nel}")
        # self.log.debug(f"N length: {nth_length}")

        lower_bound = pd.Series([data.loc[0, "temp"] + cfg[0]], index=[0])

        # Combine multiple time series into final lower bound
        for i in range(0, nel):
            min_range = i * nth_length
            # min_range = (min_range-(i+1)) if i == (nel-1) else min_range
            min_range = 0 if min_range < 0 else min_range

            max_range = ((i + 1) * nth_length) + 1
            max_range = (
                data.shape[0] - 1 if max_range > (data.shape[0] - 1) else (max_range)
            )

            # self.log.debug(f"Range: [{min_range}, {max_range}]")
            # self.log.debug(f"Value: [{data.loc[min_range, "temp"] + cfg[i]}, {data.loc[ max_range, "temp"] + cfg[i+1]}]")

            lower_bound_nth = pd.Series(
                interpolate_slope(
                    [min_range, max_range],
                    [
                        data.loc[min_range, "temp"] + cfg[i],
                        data.loc[max_range, "temp"] + cfg[i + 1],
                    ],
                )
            )
            lower_bound_nth = pd.Series(
                lower_bound_nth[1].transpose(), index=lower_bound_nth[0].transpose()
            )

            # self.log.debug(f"Bound {i} shape : {str(lower_bound_nth.shape)}")
            if lower_bound is None:
                lower_bound = lower_bound_nth
            else:
                lower_bound = lower_bound.append(lower_bound_nth, verify_integrity=True)

        lower_bound = lower_bound.append(
            pd.Series(
                [data.loc[data.shape[0] - 1, "temp"] + cfg[nel]],
                index=[data.shape[0] - 1],
            ),
            verify_integrity=True,
        )

        # lower_bound.reset_index(inplace=True)
        # lower_bound = lower_bound.reindex_like(copy=True)
        # self.log.debug(f"Full shape : {str(lower_bound.shape)}")

        if file_name is not None:
            name = f"{file_name[:-4]}_bound.csv"
            # Just check of the lower bound generation correctness
            lower_bound.to_csv(name)
            self.log.debug(f"File written {name} => {str(lower_bound.shape)}")

        return lower_bound

    def shape_lower_temp(self, create_lower_bound: object) -> None:
        """
        Method to shapes lower part of the autoclave profile of
        the signal modification to demonstrate model drift.
        The aim of this process is to showcase model drift
        by using various detector methods.
        Args:
            create_lower_bound  - function that creates lower shape
                                  of the data input
        Returns:
            <None>
        """
        cfg = self.cfg["change_val"]["linear"]

        for file_path in self.cfg["file_paths"]:
            data = pd.read_csv(file_path[0])
            data.reset_index(inplace=True)

            # Add file_path for
            lower_bound = create_lower_bound(cfg, data, None)  # , file_path[1])

            # 2. Merge signals
            # Iterate over time domain and compare signal value with
            # lower_bound values. If a value at given time is lower than
            # required it will be adjusted by the value from lower_bound
            # otherwise the value will stay unchanged.
            new_set = pd.DataFrame(data)

            # Single file time domain iteration
            for i in range(0, len(new_set)):
                if data.loc[i, "temp"] < lower_bound[i]:
                    new_set.loc[i, "temp"] = lower_bound[i]
                if data.loc[i, "tc"] < lower_bound[i]:
                    new_set.loc[i, "tc"] = lower_bound[i]

            # Write new singal back to file system
            new_set.set_index("time", inplace=True)
            new_set.drop("index", 1, inplace=True)
            new_set.to_csv(file_path[1])

            self.log.debug(f"New profile written to: {file_path[1]}")

    def run(self) -> None:
        """
        Method runs through
        Args:
            <None>
        Returns:
            <None>
        """
        # self.log.debug("Pre configuration:")
        # self.log.debug(f"{json.dumps(self.cfg, indent=4)}")

        # If the file section is empty leave
        if self.cfg["file_paths"] is None:
            return

        # Check configuration capabilities
        if (
            sub_key_value := self.cfg.get("dissect", {}).get("hike")
            and self.cfg["dissect"]["hike"] is not None
            and len(self.cfg["dissect"]["hike"]) > 0
        ):
            self.log.debug("Dissecting data:")
            self.log.debug(
                f"Config: {json.dumps(self.cfg['dissect']['hike'], indent=4)}"
            )
            self.dissect()

        if (
            sub_key_value := self.cfg.get("change_val", {}).get("linear")
            and self.cfg["change_val"]["linear"] is not None
            and len(self.cfg["change_val"]["linear"]) > 1
        ):
            self.log.debug(
                f"Modifying signal -> linear mod [{len(self.cfg['change_val']['linear'])}]]:"
            )
            self.log.debug(
                f"Config: {json.dumps(self.cfg['change_val']['linear'], indent=4)}"
            )
            self.shape_lower_temp(self.nth_linear_lbound)


def main() -> None:
    """
    Method runs simulation process
    Args:
        <None>
    Returns:
        <None>
    """
    parser = argparse.ArgumentParser(prog="ICS PRE script", usage="%(prog)s [options]")
    parser.add_argument("-c", type=open, help="Configuration file for the script")
    args = parser.parse_args()

    l2F = Log("ics_pre.log")

    l2F.debug(f"Config: {str(args.c.name)}")
    l2F.debug(args.c.name)

    # Load simulator configuration
    cfg = yc.toJson(args.c.name)

    pre = PreProcessor(cfg, l2F)
    pre.run()

    if l2F is not None:
        l2F.close()


if __name__ == "__main__":
    main()
