"""
Abstract dictionary subclass and discriptor class for accessing this dictionary.
"""
from __future__ import annotations

from collections import UserDict


class AbstractLookupDict(UserDict):
    """Abstract dictionary subclass that lazily and dynamically returns values."""
    def __init__(self, dict=None, /, **kwargs):
        super().__init__(dict)
        for attr_name, attr_value in kwargs.items():
            if hasattr(self, attr_name):
                raise AttributeError("Cannot set attribute that already exists.")
            setattr(self, attr_name, attr_value)

    def __contains__(self, key: object) -> bool:
        if hasattr(self.__class__, "__missing__"):
            try:
                self.__class__.__missing__(self, key)
            except KeyError:
                return False
        return super().__contains__(key)


class AbstractDictDescriptor:
    """Descriptor as lookup dictionary that lazily and dynamically returns values."""
    def __set_name__(self, owner, name):
        self.private_name = '_' + name


    def __get__(self, instance, _cls=None):
        if not hasattr(instance, self.private_name):
            self.init_lookup(instance)

        return getattr(instance, self.private_name)


    def __set__(self, instance, value):
        self.__delete__(instance)
        self.__get__(instance, type(instance)).update(value)


    def __delete__(self, instance):
        if hasattr(instance, self.private_name):
            delattr(instance, self.private_name)


    def init_lookup(self, instance):
        """Initialize the lookup dictionary and add it to ``instance``."""
        raise NotImplementedError("Subclasses must implement this method.")
