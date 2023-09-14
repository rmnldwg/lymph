"""
Module containing mostly descriptors that are used to dynamically construct expensive
attributes of a :py:class:`lymph.models.Unilateral` instance.
"""
from __future__ import annotations

from collections import UserDict


class AbstractLookupDict(UserDict):
    """Abstract ``UserDict`` subclass that can lazily and dynamically return values.

    This class is meant to be subclassed. If one wants to use the functionality
    of lazy and dynamic value retrieval, the subclass must implement a ``__missing__``
    method that returns the value for the given key and raises a ``KeyError`` if
    the value for a key cannot be computed.
    """
    def __init__(self, dict=None, /, trigger_callbacks=None, **kwargs):
        """Use keyword arguments to set attributes of the instance.

        In contrast to the default ``UserDict`` constructor, this one instantiates
        any keyword arguments as attributes of the instance and does not put them
        into the dictionary itself.
        """
        super().__init__(dict)

        if trigger_callbacks is None:
            trigger_callbacks = []

        kwargs.update(trigger_callbacks=trigger_callbacks)

        for attr_name, attr_value in kwargs.items():
            if hasattr(self, attr_name):
                raise AttributeError("Cannot set attribute that already exists.")
            setattr(self, attr_name, attr_value)


    def __contains__(self, key: object) -> bool:
        """This exists to trigger ``__missing__`` when checking ``is in``."""
        if hasattr(self.__class__, "__missing__"):
            try:
                self.__class__.__missing__(self, key)
            except KeyError:
                return False
        return super().__contains__(key)
