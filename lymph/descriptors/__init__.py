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
    def __init__(self, dict=None, /, **kwargs):
        """Use keyword arguments to set attributes of the instance."""
        super().__init__(dict)
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


class AbstractDictDescriptor:
    """Descriptor that constructs itself when the attribute it manages is missing.

    It expects the attribute to be dict-like and implements the ``__get__``,
    ``__set__`` and ``__delete__`` methods. When the attribute is missing, it
    calls the ``_get_callback`` method to construct the attribute. This method
    must be implemented by subclasses.

    It allows to lazily and dynamically construct the attribute it manages when the
    attribute is accessed but not present. This is useful for attributes that
    are expensive to compute and may need recomputation when the state of the
    instance changes. In that case, one only has to delete the attribute and
    the next time it is accessed, it will be recomputed automatically.
    """
    def __set_name__(self, owner, name):
        self.private_name = '_' + name


    def __get__(self, instance, _cls=None):
        if not hasattr(instance, self.private_name):
            self._get_callback(instance)

        return getattr(instance, self.private_name)


    def __set__(self, instance, value):
        self.__delete__(instance)
        self.__get__(instance, type(instance)).update(value)


    def __delete__(self, instance):
        if hasattr(instance, self.private_name):
            delattr(instance, self.private_name)


    def _get_callback(self, instance):
        """Perform lazy dynamic stuff when ``__get__`` is called."""
        raise NotImplementedError("Subclasses must implement this method.")
