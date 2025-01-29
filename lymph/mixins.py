"""Mixin classes to enhance functionality of models."""

from collections.abc import Sequence

from lymph.types import ParamsType


class NamedParamsMixin:
    """Allow defining a :py:attr:`.named_params` subset of params to set and get."""

    @property
    def named_params(self) -> Sequence[str]:
        """Sequence of parameter names that may be changed.

        Only parameter names are allowed that would also be recognized by the
        :py:meth:`~lymph.types.Model.set_params` method. For example, ``"TtoII_spread"``
        or ``"late_p"`` could be valid named parameters. Even global parameters like
        ``"spread"`` work.

        .. warning::

            The order is important: If the :py:attr:`.named_params` are set to e.g.
            ``["TtoII_spread", "spread"]``, then the ``"spread"`` parameter will
            override the ``"TtoII_spread"``.

        This exists for reproducibility reasons: It allows for a subset of parameters
        to be set via a special method (:py:meth:`.set_named_params`). Subsequently,
        only these parameters can be set via that method, both using positional and
        keyword arguments.

        A use case for this is parameter sampling. E.g., someone samples only a subset
        of parameters and stores these as an unnamed array along with a list of the
        parameters names they correspond to. Without the :py:attr:`.named_params`
        and the :py:meth:`.set_named_params` method, it would be tricky to load those
        values back into the model.

        .. seealso::

            `This issue`_ on GitHub provides more information for the rationale behind
            this mixin.

        .. _This issue: https://github.com/rmnvsl/lymph/issues/95
        """
        return getattr(self, "_named_params", self.get_params(as_dict=True).keys())

    @named_params.setter
    def named_params(self, new_names: Sequence[str]) -> None:
        """Set the named params."""
        if not isinstance(new_names, Sequence):
            try:
                new_names = list(new_names)
            except TypeError as te:
                raise ValueError("Named params must be castable to a sequence.") from te

        default_params = self.get_params(as_dict=True, as_flat=True).keys()
        joined_defaults = "|".join(default_params)

        for name in new_names:
            if not name.isidentifier():
                raise ValueError(f"Named param {name} isn't valid identifier.")
            if name not in joined_defaults:
                raise ValueError(f"Named param {name} not among settable params.")

        self._named_params = new_names

    def get_named_params(self, as_dict: bool = True) -> ParamsType:
        """Get the values of the :py:attr:`.named_params`.

        .. note::

            Unlike the general :py:meth:`~lymph.types.Model.get_params` method, this
            method does not support the keyword argument ``as_flat``. The returned
            dictionary (if ``as_dict=True``) will always be flat.
        """
        all_params = self.get_params(as_dict=True, as_flat=True)
        named_params = {k: all_params[k] for k in self.named_params}
        return named_params if as_dict else named_params.values()

    def set_named_params(self, *args, **kwargs) -> None:
        """Set the values of the :py:attr:`.named_params`."""
        new_params = dict(zip(self.named_params, args, strict=False))
        new_params.update(kwargs)
        self.set_params(**new_params)
