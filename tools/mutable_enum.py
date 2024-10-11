""" Threat factory that generates attacks on the data set (cycle) """
from __future__ import annotations


class MutableEnum(type):
    """
    Mutable enumerator that helps with dynamic enumerator class loading
    """
    values = {}

    @classmethod
    def add(cls, key: str, value: object) -> None:
        """
        Method adds new item to existing
        """
        if key not in cls.values.keys():
            cls.values[key] = value

        if not hasattr(cls, key):
            setattr(cls, key, value)

    def __getitem__(cls, key):
        return getattr(cls, key)

    def __setitem__(cls, key, value):
        setattr(cls, key, value)

    def __iter__(cls):
        # Exclude attributes starting with underscore (private in Python)
        return [key for key in cls.__dict__.keys() if not key.startswith('_')]

    def __len__(cls):
        return len(cls.__dict__.keys())

    def __contains__(cls, p):
        return True if p in cls.__dict__.keys() else False
