"""Mixin classes to enhance functionality of models."""

from collections.abc import Sequence


class NamedParamsMixin:
    """Allow defining a ``named_params`` subset of params to set via special method."""

    @property
    def named_params(self) -> Sequence[str]:
        """Get the named params."""
        return getattr(self, "_named_params", self.get_params(as_dict=True).keys())

    @named_params.setter
    def named_params(self, new_names: Sequence[str]) -> None:
        """Set the named params."""
        if not isinstance(new_names, Sequence):
            try:
                new_names = list(new_names)
            except TypeError as te:
                raise ValueError("Named params must be castable to a sequence.") from te

        default_params = self.get_params(as_dict=True).keys()
        joined_defaults = "|".join(default_params)

        for name in new_names:
            if not name.isidentifier():
                raise ValueError(f"Named param {name} isn't valid identifier.")
            if name not in joined_defaults:
                raise ValueError(f"Named param {name} not among settable params.")

        self._named_params = new_names

    def set_named_params(self, *args, **kwargs) -> None:
        """Set named params."""
        new_params = dict(zip(self.named_params, args, strict=False))
        new_params.update(kwargs)
        self.set_params(**new_params)
