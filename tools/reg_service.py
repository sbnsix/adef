from __future__ import annotations
import threading
from log import Log


class RegService:
    """
    Class implements wrapper for the registry service used in other
    classes inside ADEF framework
    """

    def __init__(self, logger: Log) -> None:
        """
        CTOR
        Args:
            logger: logger object
        Returns:
             <None>
        """
        self.log = logger
        # Thread save registry service
        self.mutex = threading.Lock()
        self.reg = {}

    def add(self, name: str, func: object) -> None:
        """
        Method add service into registry
        Args:
            name:
            func:
        Returns:
            <None>
        """
        with self.mutex:
            self.reg[name] = func

    def rem(self, name: str, func: object) -> None:
        """
        Method add service into registry
        Args:
            name:
            func:
        Returns:
            <None>
        """
        with self.mutex:
            if name in self.reg.keys():
                del self.reg[name]

    def get(self) -> dict:
        """
        Returns all registered services.
        Returns:
            <dict>
        """
        with self.mutex:
            return self.reg
