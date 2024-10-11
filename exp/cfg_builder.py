""" Experiment builder helper and default experiment configuration """

import os
import sys
import glob

PATHS = ["./", "../tools"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

import yaml_converter as yc
from intr import Interpreter

from log import Log


class CfgBuilderHelper:
    """
    Class encapsulating helper functions used in the configuration builder
    """

    @staticmethod
    def get_files(search_mask: str, skip_mask: str = None) -> list:
        """
        Method finds
        Args:
            search_mask: file search mask in given directory - including
            skip_mask: file to skip if search mask is broad
        Returns:
            list<str> - list of existing file name matching search criteria
        """
        if skip_mask:
            files = [
                file for file in glob.glob(search_mask) if not file.endswith(skip_mask)
            ]
        else:
            files = glob.glob(search_mask)

        if files:
            return files

        return []

    @staticmethod
    def get_file(search_mask: str, skip_mask: str = None) -> str:
        """
        Method finds
        Args:
            search_mask: file search mask in given directory - including
            skip_mask: file to skip if search mask is broad
        Returns:
            <str> - existing file name matching search criteria
        """
        files = CfgBuilderHelper.get_files(search_mask, skip_mask)
        if files:
            return files[0]

        return ""


class CfgBuilder:
    """
    Factory defining configuration modifications for the
    given experiments
    """

    @staticmethod
    def default(cfg: dict, log: Log, scafold: bool = True) -> dict:
        """
        Converts default configuration to single experiment configuration
        Args:
            cfg : high level configuration input in form of JSON
            log : logger object (used for debugging new experiments configuration builds)
            scafold : flag to enable creation of experiment directory root
        Returns:
            <dict> - JSON configuration setup ready for use in single experiment
        """
        # Load all required default configuration
        def_cfg = {}
        for file in glob.glob("./cfg/main/*.yml"):
            def_cfg |= yc.toJson(file)

        n_cfg = def_cfg | {"experiment": cfg, "global": cfg["global"]}

        # if n_cfg["experiment"]["path"] != f"{n_cfg['global']['path']}/experiments/{n_cfg['experiment']['path']}":
        #    n_cfg["experiment"]["path"] = f"{n_cfg['global']['path']}/experiments/{n_cfg['experiment']['path']}"

        n_cfg["experiment"]["path"] = n_cfg["experiment"]["path"].replace("//", "/")
        if not n_cfg["experiment"]["path"].endswith("/"):
            n_cfg["experiment"]["path"] += "/"

        if scafold:
            local_path = f"{n_cfg['global']['path']}experiments/{n_cfg['experiment']['path']}"
            Interpreter.run(
                [
                    f"md {local_path}",
                    f"md {local_path}/attack",
                    f"md {local_path}/input",
                    f"md {local_path}/detection",
                    f"md {local_path}/error",
                    f"md {local_path}/model",
                    f"md {local_path}/graph",
                    f"md {local_path}/output",
                    f"md {local_path}/timing",
                    f"md {local_path}/checkpoint",
                    f"cp {n_cfg['global']['path']}input/*.csv {local_path}/input/",
                    f"cp {n_cfg['global']['path']}attack/*.csv {local_path}/attack/",
                    f"cp {n_cfg['global']['path']}attack/*.png {local_path}/attack/",
                ]
            )

        return n_cfg
