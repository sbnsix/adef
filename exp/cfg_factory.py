""" Experiment configuration factory """
from __future__ import annotations

import gc

from log import Log
from auto_loader import AutoLoader

class CfgBuilderFactory:
    """
    Factory defining configuration modifications for the
    given experiments
    """

    # Automatic map of defined experiment configurations
    map = {}

    @staticmethod
    def get(exp_name: str, cfg: dict, logger: Log, *args) -> list:
        """
        Generator method to yield single experiment configurations
        Args:
            exp_name: experiment name
            cfg : initial high level configuration that needs to be resolved
            logger : logger object (used for debugging new experiment configuration builds)
        Returns:
            <list> - list of generated configurations of atomic experiment
        Returns:
            <list> - list of configurations required for the given experiment to be correctly completed
        """
        gc.collect()
        exp_name = f"exp{exp_name.replace('_', '')}"

        if exp_name not in CfgBuilderFactory.map.keys():
            CfgBuilderFactory.map = AutoLoader.load("./exp", "exp_*.py", [cfg, logger])
        gc.collect()

        if not cfg:
            return []

        avail_tests = [
            x.replace("_", "").lower() for x in list(CfgBuilderFactory.map.keys())
        ]
        if f"exp{cfg['id'].lower()}" not in avail_tests:
            return []

        if not hasattr(CfgBuilderFactory.map[exp_name], "get"):
            return []

        cfgs = CfgBuilderFactory.map[exp_name].get(args)

        gc.collect()

        return cfgs

    @staticmethod
    def post(exp_name: str, cfg: dict, logger: Log, *args) -> None:
        """
        Generator method to run post configuration on given experiment
        Args:
            exp_name: experiment name
            cfg : initial high level configuration that needs to be resolved
            logger : logger object (used for debugging new experiment configuration builds)
        Returns:
            <list> - list of generated configurations of atomic experiment
        Returns:
            <list> - list of configurations required for the given experiment to be correctly completed
        """

        gc.collect()

        if not cfg:
            return

        exp_name = f"exp{exp_name.replace('_', '')}"

        avail_tests = [x.lower() for x in list(CfgBuilderFactory.map.keys())]
        if f"exp{cfg['id'].lower()}" not in avail_tests:
            return

        if not hasattr(CfgBuilderFactory.map[exp_name], "post"):
            return

        result = CfgBuilderFactory.map[exp_name].post(cfg, logger, args)

        gc.collect()

        return result
