"""Provide a descriptor class to access parameters of a lymph model."""

from typing import Callable


class Param:
    """Stores getter and setter functions for a parameter."""
    def __init__(self, model, getter: Callable, setter: Callable):
        self.model = model
        self.get = getter
        self._set = setter

    def set(self, value):
        """Delete the transition matrix when setting a parameter."""
        self._set(value)
        del self.model.transition_matrix


class ParamDict(dict):
    """Dictionary preventing direct setting of parameter values."""
    def __setitem__(self, __key: str, __value: Param) -> None:
        if not isinstance(__value, Param):
            raise TypeError(
                "Params cannot be accessed directly! "
                "Use the `get` & `set` methods instead."
            )
        return super().__setitem__(__key, __value)


class Lookup:
    """Descriptor class to access parameters of a lymph model.

    When first trying to access this descriptor, it will compute a lookup table
    for all parameters of the model. This is done by iterating over all edges
    and creating a dictionary with keys of parameter names and values of `Param`
    objects. These Param objects store the getter and setter functions for the
    corresponding parameter.
    """
    def __get__(self, instance, _cls) -> ParamDict:
        """Test"""
        if not hasattr(self, "lookup"):
            self._init_params_lookup(instance)
        return self.lookup

    def _init_params_lookup(self, instance):
        self.lookup = ParamDict()

        for edge in instance.tumor_edges:
            self.lookup['spread_' + edge.name] = Param(
                model=instance,
                getter=edge.get_spread_prob,
                setter=edge.set_spread_prob,
            )

        for edge in instance.lnl_edges:
            self.lookup['spread_' + edge.name] = Param(
                model=instance,
                getter=edge.get_spread_prob,
                setter=edge.set_spread_prob,
            )

            if instance.is_trinary:
                self.lookup['micro_' + edge.name] = Param(
                    model=instance,
                    getter=edge.get_micro_mod,
                    setter=edge.set_micro_mod,
                )

        # here we don't need to check if the model is trinary, because the growth edges
        # are only present in trinary models
        for edge in instance.growth_edges:
            self.lookup['growth_' + edge.start.name] = Param(
                model=instance,
                getter=edge.get_spread_prob,
                setter=edge.set_spread_prob,
            )
