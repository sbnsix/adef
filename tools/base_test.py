""" Base test class for all unit tests """


from __future__ import annotations

import os
import sys
import logging
import unittest

PATHS = ["./"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if not path in sys.path:
        sys.path.insert(1, path)

import log
import file_util as fu


class BaseTest(unittest.TestCase):
    """
    CSV Test Class that validates casWrapper interface
    """

    @classmethod
    def setUpClass(cls, file_name: str) -> None:
        """
        Setup CSV connection
        Args:
            fileName    -   name of the log file
        Returns:
            <BaseTest>  - instance of class BaseTest
        """
        super().setUpClass()
        log_file = "./log/" + file_name + ".log"
        fu.FileUtil.remove(log_file)

        cls.log = log.Log(file_name, log_file, logging.DEBUG)

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Test method closes backend connection
        Args:
            <None>
        Returns:
            <None>
        """
        super().tearDownClass()
        # Close logger
        if cls.log is not None:
            cls.log.close()
