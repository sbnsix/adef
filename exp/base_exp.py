"""Experiment configuration base class"""

from __future__ import annotations
import os
import sys

from cfg_builder import CfgBuilder

PATHS = ["./", "../tools", "../pres/ppt", "../pres/latex"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from log import Log


class BaseExperimentConfig:
    def __init__(self, cfg: dict, logger: Log) -> None:
        """
        CTOR
        Args:
            cfg : high level configuration input in form of JSON
            logger : logger object (used for debugging new experiment configuration builds)
        Returns:
            <None>
        """
        self.log = logger
        self.cfg = cfg
        self.d_cfg = CfgBuilder.default(cfg, logger, True)

    def get(self) -> list:
        raise NotImplementedError(
            "Please add derived get method for inherited class of BaseExperimentConfig"
        )

    def post(self) -> None:
        pass
