""" ADEF meta configuration automation script """
from __future__ import annotations

import os
import json

import subprocess
import sys
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ---------------------------------------------
#                Local imports
# ---------------------------------------------
PATHS = ["./", "./tools", "./exp"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from tools.log import Log
import tools.yaml_converter as yc
from tools.os_utility import Utility

# Local includes
from tools.experiment import Experiment

from exp.cfg_factory import CfgBuilderFactory
from tools.intr import Interpreter
from mev.data.attack_scenarios import DataGeneration
from tools.cfg_helper import ConfigHelper


class CPSMeta:
    """
    Class defining behaviour of the automation script
    that will run automatically series of experiments
    """

    def __init__(self, cfg: dict, logger: Log) -> None:
        """
        CTOR
        Args:
            cfg     - configuration object
            logger  - logger object
        Returns:
            <CPS_Meta>  - instance of CPS_Meta object
        """
        self.cfg = cfg
        self.log = logger
        self.id = "meta_level"
        self.shutdown = cfg["global"]["autoshutdown"]
        self.cycle_loop = None

        # Preload Process, Attack, Model and Results sections
        self.cfg_process = yc.toJson("./cfg/main/process.yml")
        self.cfg_attack = yc.toJson("./cfg/main/attack.yml")
        self.cfg_model = yc.toJson("./cfg/main/model.yml")
        self.cfg_results = yc.toJson("./cfg/main/results.yml")

        self.m_path = ""
        self.o_path = ""
        self.a_path = ""

        # Main path of the experiment
        self.main_path = ""

    def __pre(self, cfg: dict) -> (bool, dict):
        """
        Meta script performs pre-configuration tasks
        i.e. auto generation/aggregation tasks form
        all experiments that will be executed in the
        framework.
        Args:
            cfg - local experiment configuration
        Returns:
            <bool> - True to run PRE initialization of the experiment, otherwise False
        """
        # TODO: Add all four automatic configuration tasks for each experiment.
        #  this might be execution of multiple functions - each responsible for
        #  single experiment configuration.
        result = True

        # The ICS trace and attacks are used across all type of experiments to have
        # the same base-line. Some elements are changed but this change will this
        # data set to adjust settings to given experiment.
        new_path = f"{cfg['global']['path']}"
        curr_path = os.getcwd().replace("\\", "/")
        Interpreter.run(
            [
                f"md {new_path}",
                f"md {new_path}/cfg",
                f"md {new_path}/cfg/main",
                f"md {new_path}/cfg/env",
                f"cp {curr_path}/cfg/global_config.yml {new_path}/cfg",
                f"cp {curr_path}/cfg/ad_config.py {new_path}/cfg",
                f"cp {curr_path}/cfg/main/*.yml {new_path}/cfg/main",
                f"cp {curr_path}/cfg/env/*.csv {new_path}/cfg/env",
                f"md {new_path}/attack",
                f"md {new_path}/input",
                f"md {new_path}/checkpoint",
                f"md {new_path}/timing",
                f"md {new_path}/experiments",
            ],
            self.log,
        )

        cfg = ConfigHelper.load(f"{cfg['global']['path']}cfg/global_config.yml")

        # Generate ICS system profile
        DataGeneration.step_1(cfg, self.log)

        # Generate attacks based on the main configuration
        DataGeneration.step_2(cfg, self.log)

        self.log.debug("=" * 20 + "PRE simulation" + "=" * 20)

        return result, cfg

    def run(self, cfg: dict) -> None:
        """
        Meta script method to perform
        Args:
            cfg: experiment configuration
        Returns:
            <None>
        """
        # Run all preconditions for experiment
        # (data + attack generation)
        run_pre, exp_cfg = self.__pre(cfg)

        if not run_pre:
            self.log.error("PRE configuration of the meta level experiment FAILED.")
            return

        self.log.debug("=" * 20 + "Simulation" + "=" * 20)
        start = datetime.now()

        for exp_name, exp_pack in cfg["experiments"].items():
            self.log.debug(f"Experiment: {exp_name} => START")
            experiment_cfg = (
                {"global": cfg["global"]} | {"post": cfg["post"]} | exp_pack
            )

            exp_cfgs = CfgBuilderFactory.get(exp_name, experiment_cfg, self.log, True)

            # Experiment configuration factory generates single configuration
            for local_cfg in exp_cfgs:
                # Run here all single experiments
                # self.log.debug(f"Single EXP CFG:\n |{json.dumps(local_cfg, indent=4)}|")

                exp = Experiment(local_cfg, self.log)
                exp.run()

            # POST experiment data collection (summary for whole experiment)
            CfgBuilderFactory.post(
                exp_name.lower(), experiment_cfg | {"cfgs": exp_cfgs}, self.log
            )

            self.log.debug(f"Experiment: {exp_name} => STOP")

        self.log.debug(
            f"All tests completed in {(datetime.now() - start).total_seconds():.2f} seconds"
        )

        # Summary slides
        self.__post(exp_cfg)

    def __post(self, cfg: dict) -> None:
        """
        Meta script performs post configuration tasks
        i.e. auto generation/aggregation tasks form
        all experiments that were executed
        Args:
            cfg - local experiment configuration
        Returns:
            <None>
        """
        self.log.debug("=" * 20 + " POST simulation " + "=" * 20)

        if cfg["post"] is None:
            return

        if "post" not in cfg.keys():
            return

        # Execute all required post-configuration scripts
        for cmd in cfg["post"]:
            cmds = cmd.split(" ")
            if os.path.isfile(cmds[0]):
                try:
                    cdx = ["python.exe"] + cmds
                    subprocess.run(cdx, check=False)
                except Exception as ex:
                    self.log.exception(ex)
            else:
                self.log.error(f"File doesn't exists {cmds[0]}. Cannot run the script!")

        # Perform auto-shutdown
        if self.shutdown:
            Utility.auto_shutdown(datetime.now(), self.log)


def main():
    """
    Main entry to the ADEF framework simulation
    Returns:
        <int>: result of the script execution
    """
    l2f = Log("adef.log")

    # Load simulator configuration
    cfg = yc.toJson("./cfg/global_config.yml")

    meta_experiment = CPSMeta(cfg, l2f)
    meta_experiment.run(cfg)

    if l2f is not None:
        l2f.close()


if __name__ == "__main__":
    main()
