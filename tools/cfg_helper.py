""" Configuration helper module """

from __future__ import annotations
import os
import sys
import glob

PATHS = ["./", "../tools"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

import os.path
import yaml_converter as yc


class ConfigHelper:
    """
    Configuration helper class that loads and combines configuration
    from all *.yml files for ADEF framework.
    """

    @staticmethod
    def load(cfg_file: str) -> dict:
        """
        Method load simulator configuration
        Args:
            cfg_file: name of the configuration file
        Returns:
            <dict> - updated configuration file
        """
        if not os.path.isfile(cfg_file):
            raise FileNotFoundError(f"File: {cfg_file} doesn't exists")
        cfg = yc.toJson(cfg_file)

        main_path = cfg_file

        idx = main_path.rfind("/")
        if idx > 0:
            main_path = main_path[:idx]

        for add_cfg in glob.glob(f"{main_path}/main/*.yml"):
            if not os.path.isfile(add_cfg):
                raise FileNotFoundError(f"File main: {add_cfg} doesn't exists")

            cfg.update(yc.toJson(add_cfg))

        cfg["experiment"] = {"path": cfg["global"]["path"]}

        return cfg
