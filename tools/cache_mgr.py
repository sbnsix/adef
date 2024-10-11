""" Cache module  """

from log import Log
import threading

# Read those
# https://pypi.org/project/cachetools/
# https://realpython.com/python-memcache-efficient-caching/


class CacheManager:
    """
    Class implementing caching mechanism to accelerate I/O handling
    in the simulation engine.
    """

    def __init__(self, log: Log) -> None:
        """
        CTOR
        Args:
            log: logger object
        Returns:
            <None>
        """
        # Dictionary element containing the in memory stored elements
        self.mutex = threading.Lock()
        self.cache = {}

    def get(self, key: str) -> object:
        """
        Method fetches object from the cache
        Args:
            key: key identifier through which stored object is saved in memory
        Returns:
            <object> - cached object
        """
        with self.mutex:
            if key in self.cache.keys():
                return self.cache[key]
        return None

    def put(self, key: str, obj: object) -> None:
        """
        Method puts an object to internal cache
        Args:
            key: key identifier used to identify object
            obj: object to be stored in cache
        Returns:
            <None>
        """
        with self.mutex:
            self.cache[key] = obj

    def clear(self) -> None:
        """
        Method clear cache
        Returns:
            <None>
        """
        self.cache = {}
