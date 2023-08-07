"""
Abstract dictionary subclass and discriptor class for accessing this dictionary.
"""
from lymph import models


class AbstractLookupDict(dict):
    """Abstract dictionary base class created and managed by subclasses of the
    :py:class:`AbstractLookup`.

    Its primary use is to store a reference to the model it belongs to. Also, it
    ensures that the ``update()`` method calls ``__setitem__()``.
    """
    def __init__(self, model: models.Unilateral, **kwargs):
        super().__init__()
        self.update(**kwargs)
        self.model = model


    def update(self, new_dict: dict):
        for key, value in new_dict.items():
            self[key] = value


class AbstractLookup:
    """Generic decriptor class to access and initialize lookup dictionaries.

    The implementation is kept simple: If the attribute is accessed, but doesn't exist,
    the method ``init_lookup`` is called and passed the instance of the model class.
    In case someone tries to set the attribute with a new dictionary, the lookup
    dictionary is deleted and replaced entirely with a new one, which is created
    empty when calling ``__get__`` and then updated.
    """
    def __set_name__(self, owner, name):
        self.private_name = '_' + name


    def __get__(self, instance, _cls):
        if not hasattr(instance, self.private_name):
            self.init_lookup(instance)

        return getattr(instance, self.private_name)


    def init_lookup(self, model: models.Unilateral):
        """Initialize the lookup dictionary. Must be implemented by subclasses."""
        raise NotImplementedError


    def __set__(self, instance, value):
        self.__delete__(instance)
        self.__get__(instance, type(instance)).update(value)


    def __delete__(self, instance):
        if hasattr(instance, self.private_name):
            delattr(instance, self.private_name)
