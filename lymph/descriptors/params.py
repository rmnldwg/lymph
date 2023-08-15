"""Provide a descriptor class to access parameters of a lymph model."""
from __future__ import annotations

from lymph import graph
from lymph.descriptors import AbstractDictDescriptor, AbstractLookupDict


class Param:
    """Stores getter and setter functions for a parameter.

    This simple class also makes sure that the transition matrix of the model
    is deleted when a spread parameter is set.
    """
    def __init__(self, getter: callable, setter: callable):
        self.get_param = getter
        self._set_param = setter

    def set_param(self, value):
        """Delete the transition matrix when setting a parameter."""
        self._set_param(value)


class ParamsUserDict(AbstractLookupDict):
    """Dictionary class to store the parameters of a lymph model."""
    def __setitem__(self, key: str, value: Param, / ) -> None:
        if not isinstance(value, Param):
            raise TypeError(
                "Params cannot be accessed directly! "
                "Use the `get_param` & `set_param` methods instead."
            )
        return super().__setitem__(key, value)


class GetterSetterAccess(AbstractDictDescriptor):
    """Descriptor class to access parameters of a lymph model.

    When first trying to access this descriptor, it will compute a lookup dictionary
    for all parameters of the model. This is done by iterating over all edges
    and creating a dictionary with keys of parameter names and values of `Param`
    objects. These Param objects store the getter and setter functions for the
    corresponding parameter.
    """
    def _get_callback(self, instance: graph.Representation):
        """Compute the lookup table for all edge parameters of the lymph model."""
        params_dict = ParamsUserDict()
        for edge in instance._tumor_edges:
            params_dict['spread_' + edge.name] = Param(
                getter=edge.get_spread_prob,
                setter=edge.set_spread_prob,
            )

        for edge in instance._lnl_edges:
            params_dict['spread_' + edge.name] = Param(
                getter=edge.get_spread_prob,
                setter=edge.set_spread_prob,
            )

            if instance.is_trinary:
                params_dict['micro_' + edge.name] = Param(
                    getter=edge.get_micro_mod,
                    setter=edge.set_micro_mod,
                )

        # here we don't need to check if the model is trinary, because the growth edges
        # are only present in trinary models
        for edge in instance._growth_edges:
            params_dict['growth_' + edge.parent.name] = Param(
                getter=edge.get_spread_prob,
                setter=edge.set_spread_prob,
            )

        setattr(instance, self.private_name, params_dict)
