""" ICS model data generation script """

from __future__ import annotations

import os
import sys
import pandas as pd
import glob

# ---------------------------------------------
#                Local imports
# ---------------------------------------------
PATHS = ["./", "../../icsr/models"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)
from tools.log import Log
from tools.checkpoint import checkpoint
from tools.checkpoint import timing

from autoclave.model.process import AutoclaveProcess
import attacker
from attack_gen import AttackScenario


class DataGeneration:
    """
    Data generation part of the simulation
    """

    @staticmethod
    @timing
    @checkpoint("step1")
    def step_1(cfg: dict, logger: Log) -> None:
        """
        Method performs STEP.1. Cyber-physical process data generation -
        autoclave process simulation of ADEF experiment. It generates autoclave process data.
        Args:
            cfg: meta level configuration object
            logger: logger object
        Returns:
            <None>
        """

        logger.debug("STEP.1 - Data generation")
        p_path = cfg["experiment"]["path"]
        p_path += cfg["process"]["path"]

        if cfg["process"]["profile"]["enabled"]:
            if not os.path.isdir(p_path):
                os.mkdir(p_path)

            if (
                not os.path.isfile(f"{p_path}{cfg['process']['profile']['file']}")
                or cfg["process"]["profile"]["enabled"]
            ):
                params, data = AutoclaveProcess.gen_data(cfg, logger)
                data.to_csv(f"{p_path}{cfg['process']['profile']['file']}", index=False)
            else:
                data = pd.read_csv(f"{p_path}{cfg['process']['profile']['file']}")

            # Modify data to mirror for differential manifold
            data_mir = data.copy(deep=True)
            data_mir.loc[:, "temp"] *= -1
            data_mir.loc[:, "tc"] *= -1
            data_mir.to_csv(
                f"{p_path}{cfg['process']['profile']['file'][:-4]}_mir.csv", index=False
            )
        # TODO: Add external file import option (experimental)
        # else:
        #    log.debug(f"Reading data file: {path}{cfg['process']['profile']['file']}")
        #    data = pd.read_csv(f"{p_path}{cfg['process']['profile']['file']}")
        #    data_mir = pd.read_csv(f"{p_path}{cfg['process']['profile']['file'][:-4]}_mir.csv")

    @staticmethod
    @timing
    @checkpoint("step2")
    def step_2(cfg: dict, logger: Log) -> None:
        """
        Method performs step 2 (the cyber-physical attack) of ADEF experiment.
        It generates autoclave process attack data in accordance to
        attack configuration for the experiment.
        Args:
            cfg - meta level configuration object

        Returns:
            <None>
        """
        data_file = [
            file
            for file in glob.glob(f"{cfg['experiment']['path']}input/*.csv")
            if not file.endswith("_mir.csv")
        ][0]
        data = pd.read_csv(data_file)
        data.set_index("time", inplace=True)

        data_mir_file = glob.glob(f"{cfg['experiment']['path']}input/*_mir.csv")[0]
        data_mir = pd.read_csv(data_mir_file)
        data_mir.set_index("time", inplace=True)

        logger.debug("STEP.2 - Attack generation")
        a_path = cfg["experiment"]["path"]
        a_path += cfg["attack"]["path"]

        if not os.path.isdir(a_path):
            os.mkdir(a_path)

        att_gen = attacker.ICS_Attacker(cfg, logger)

        if cfg["attack"]["enabled"]:
            densities = cfg["attack"]["densities"]

            for density in densities:
                enabled_attacks = {
                    key: value
                    for (key, value) in cfg["attack"]["types"].items()
                    if value["enabled"]
                }

                # Single attacks
                if cfg["attack"]["single"]:
                    AttackScenario.single_attack(
                        enabled_attacks,
                        a_path,
                        density,
                        att_gen,
                        data,
                        data_mir,
                        cfg,
                        logger,
                    )

                # One off mixed approach with known (selected attacks) and
                # unknown attacks that will be added to test/validation data
                if cfg["attack"]["mixed"]:
                    AttackScenario.mixed_attack(
                        enabled_attacks,
                        a_path,
                        density,
                        att_gen,
                        data,
                        data_mir,
                        cfg,
                        logger,
                    )
