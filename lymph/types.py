"""Type aliases and protocols used in the lymph package."""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, Protocol, TypeVar

import numpy as np
import pandas as pd


def does_contain_in_order(sequence: Sequence, items: Sequence) -> bool:
    """Check if ``sequence`` contains ``items`` in the same order (gaps allowed).

    >>> does_contain_in_order(["ipsi", "TtoII", "spread"], ["ipsi", "spread"])
    True
    >>> does_contain_in_order(["ipsi", "TtoII", "spread"], ["spread", "ipsi"])
    False
    """
    if not items:
        return True

    if not sequence:
        return False

    if sequence[0] == items[0]:
        return does_contain_in_order(sequence[1:], items[1:])

    return does_contain_in_order(sequence[1:], items)


def search_nested(
    mapping: Mapping,
    keys: Sequence[str],
    raise_keyerror: bool = True,
) -> list[float]:
    """Search a nested mapping given a sequence of keys.

    The first of the ``keys`` is used to access the first level of the mapping. If no
    such key is found, it searches all the values of the first level if they have the
    first of the ``keys``. Returns all matching values.

    If nothing is found a KeyError is raised unless ``raise_keyerror`` is ``False``.

    >>> nested = {"a": {"x": 0.1, "y": 0.2, "z": 0.3}, "b": {"x": 0.4}}
    >>> search_nested(nested, ["b", "x"])
    [0.4]
    >>> search_nested(nested, ["x"])
    [0.1, 0.4]
    >>> search_nested(nested, ["z"])
    [0.3]
    >>> search_nested(nested, ["c"])
    Traceback (most recent call last):
    ...
    KeyError: 'c'
    >>> search_nested(nested, ["c"], raise_keyerror=False)
    []
    >>> search_nested(nested, ["a", "x", "foo"])
    Traceback (most recent call last):
    ...
    TypeError: Expected `Mapping`, but got mapping=0.1. Too many keys?
    """
    if not isinstance(mapping, Mapping) and len(keys) == 0:
        return [mapping]

    if not isinstance(mapping, Mapping):
        raise TypeError(f"Expected `Mapping`, but got {mapping=}. Too many keys?")

    if len(keys) == 0:
        raise ValueError("No keys provided.")

    if keys[0] in mapping:
        return search_nested(mapping[keys[0]], keys[1:])

    results = []
    for value in mapping.values():
        try:
            results.extend(search_nested(value, keys))
        except (TypeError, KeyError):
            continue

    if len(results) == 0 and raise_keyerror:
        raise KeyError(keys[0])

    return results


class DataWarning(UserWarning):
    """Warnings related to potential data issues."""


class InvalidParamNameWarning(UserWarning):
    """Issues when an invalid parameter name is used."""


class ExtraParamsError(Exception):
    """Exception raised when additional unrecognized parameters are passed."""

    def __init__(self, extra_param_names: set[str]) -> None:
        """Initialize the exception with the extra parameter names."""
        self.invalid_param_names = extra_param_names
        self.message = f"Additional unrecognized parameter names: {extra_param_names}"
        super().__init__(self.message)


def create_alias_map(
    all_params: Iterable[str],
    named_params: Iterable[str],
) -> dict[str, list[str]]:
    """Create a mapping from named params to valid param names.

    >>> all_params = ["TtoII_spread", "TtoIII_spread", "IItoIII_spread", "late_p"]
    >>> named_params = ["spread", "TtoIII_spread"]
    >>> create_alias_map(all_params, named_params)   # doctest: +NORMALIZE_WHITESPACE
    {'spread': ['TtoII_spread', 'TtoIII_spread', 'IItoIII_spread'],
     'TtoIII_spread': ['TtoIII_spread']}
    """
    param_aliases = {}

    for named_param in named_params:
        param_aliases[named_param] = []
        for param in all_params:
            if does_contain_in_order(
                sequence=param.split("_"),
                items=named_param.split("_"),
            ):
                param_aliases[named_param].append(param)

    return param_aliases


def reverse_alias_map(aliases: dict[str, Sequence[str]]) -> dict[str, str]:
    """Reverse mapping from param aliases to valid param names.

    >>> aliases = {
    ...     "spread": ["TtoII_spread", "TtoIII_spread", "IItoIII_spread"],
    ...     "TtoIII_spread": ["TtoIII_spread"],
    ... }
    >>> reverse_alias_map(aliases)   # doctest: +NORMALIZE_WHITESPACE
    {'TtoII_spread': 'spread',
     'TtoIII_spread': 'TtoIII_spread',
     'IItoIII_spread': 'spread'}
    """
    return {alias: name for name, alias_list in aliases.items() for alias in alias_list}


class HasSetParams(Protocol):
    """Protocol for classes that have a ``set_params`` method."""

    def set_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the class."""
        ...


class HasGetParams(Protocol):
    """Protocol for classes that have a ``get_params`` method."""

    def get_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> tuple[float] | dict[str, float]:
        """Return the parameters of the class."""
        ...


GraphDictType = dict[tuple[str, str], list[str]]
"""Type alias for a graph dictionary.

A dictionary of this form specifies the structure of the underlying graph. Example:

>>> graph_dict = {
...     ("tumor", "T"): ["I", "II", "III"],
...     ("lnl", "I"): ["II"],
...     ("lnl", "II"): ["III"],
...     ("lnl", "III"): [],
... }
"""

ParamsType = Iterable[float] | dict[str, float]
"""Type alias for how parameters are passed around.

This is e.g. the type that the :py:meth:`.Model.get_params` method returns.
"""

InvolvementIndicator = Literal[
    False,
    0,
    "healthy",
    True,
    1,
    "involved",
    "micro",
    "macro",
    "notmacro",
]
"""Type alias for how to encode lymphatic involvement for a single lymph node level.

The choices ``"micro"``, ``"macro"``, and ``"notmacro"`` are only relevant for the
trinary models.
"""

PatternType = dict[str, InvolvementIndicator | None]
"""Type alias for an involvement pattern.

An involvement pattern is a dictionary with keys for the lymph node levels and values
for the involvement of the respective lymph nodes. The values are either True, False,
or None, which means that the involvement is unknown.

TODO: Document the new possibilities to specify trinary involvment.
See :py:func:`.matrix.compute_encoding`

>>> pattern = {"I": True, "II": False, "III": None}
"""

DiagnosisType = dict[str, PatternType]
"""Type alias for a diagnosis, which is an involvement pattern per diagnostic modality.

>>> diagnosis = {
...     "CT": {"I": True, "II": False, "III": None},
...     "MRI": {"I": True, "II": True, "III": None},
... }
"""


ModelT = TypeVar("ModelT", bound="Model")


class Model(ABC):
    """Abstract base class for models.

    This class provides a scaffold for the methods that any model for lymphatic
    tumor progression should implement.
    """

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

        default_params = list(self.get_params(as_dict=True, as_flat=True).keys())

        for name in new_names:
            if not name.isidentifier():
                raise ValueError(f"Named param {name} isn't valid identifier.")

            is_valid = False
            for default_name in default_params:
                if does_contain_in_order(
                    sequence=default_name.split("_"),
                    items=name.split("_"),
                ):
                    is_valid = True

            if not is_valid:
                warnings.warn(
                    message=(
                        f"Named param {name} is not a valid parameter name. "
                        "This may lead to errors during getting/setting the parameters."
                    ),
                    category=InvalidParamNameWarning,
                )

        self._named_params = new_names

    @named_params.deleter
    def named_params(self) -> None:
        """Delete the named params."""
        del self._named_params

    @abstractmethod
    def get_params(
        self: ModelT,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> ParamsType:
        """Return the parameters of the model.

        The parameters are returned as a dictionary if ``as_dict`` is True, and as
        an iterable of floats otherwise. The argument ``as_flat`` determines whether
        the returned dict is flat or nested. This is helpful, because a model may call
        the ``get_params`` method of other instances, which can be fused to get a
        flat dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def set_params(self: ModelT, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model.

        The parameters may be passed as positional or keyword arguments. The positional
        arguments are used up one by one by the ``set_params`` methods the model calls.
        Keyword arguments override the positional arguments.
        """
        raise NotImplementedError

    def get_named_params(self, as_dict: bool = True) -> ParamsType:
        """Get the values of the :py:attr:`.named_params`.

        .. note::

            Unlike the general :py:meth:`~lymph.types.Model.get_params` method, this
            method does not support the keyword argument ``as_flat``. The returned
            dictionary (if ``as_dict=True``) will always be flat.
        """
        all_params = self.get_params(as_dict=True)
        param_aliases = create_alias_map(
            all_params=all_params.keys(),
            named_params=self.named_params,
        )

        reversed_aliases = reverse_alias_map(param_aliases)
        named_params = {
            alias: all_params[param] for param, alias in reversed_aliases.items()
        }
        return named_params if as_dict else named_params.values()

    def set_named_params(self, *args, **kwargs) -> None:
        """Set the values of the :py:attr:`.named_params`.

        .. note::

            Positional arguments are overwritten by keyword arguments, which must only
            contain keys that are in :py:attr:`.named_params`.
        """
        if not set(self.named_params).issuperset(kwargs.keys()):
            extra = set(kwargs.keys()) - set(self.named_params)
            raise ExtraParamsError(extra_param_names=extra)

        new_params = dict(zip(self.named_params, args, strict=False))
        new_params.update(kwargs)
        self.set_params(**new_params)

    def get_num_dims(self: ModelT) -> int:
        """Return the number of dimensions of the parameter space.

        This is either the total number of settable parameters in the model or - if
        specified - the number of :py:attr:`.named_params`.
        """
        return len(self.get_named_params())

    @abstractmethod
    def state_dist(
        self: ModelT,
        t_stage: str,
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Return the prior state distribution of the model.

        The state distribution is the probability of the model being in any of the
        possible hidden states.
        """
        raise NotImplementedError

    def obs_dist(
        self: ModelT,
        given_state_dist: np.ndarray | None = None,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Return the distribution over observational states.

        If ``given_state_dist`` is ``None``, it will first compute the
        :py:meth:`.state_dist` using the arguments ``t_stage`` and ``mode`` (which are
        otherwise ignored). Then it multiplies the distribution over (hidden) states
        with the specificity and sensitivity values stored in the model (see
        :py:meth:`.modalities.Composite`) and marginalizes over the hidden states.
        """
        raise NotImplementedError

    @abstractmethod
    def load_patient_data(
        self: ModelT,
        patient_data: pd.DataFrame,
    ) -> None:
        """Load patient data in `LyProX`_ format into the model.

        .. _LyProX: https://lyprox.org/
        """
        raise NotImplementedError

    @abstractmethod
    def likelihood(
        self: ModelT,
        given_params: ParamsType | None = None,
        log: bool = True,
    ) -> float:
        """Return the likelihood of the model given the parameters.

        The likelihood is returned in log space if ``log`` is True, and in linear space
        otherwise. The parameters may be passed as positional or keyword arguments.
        They are then passed to the :py:meth:`set_params` method first.
        """
        raise NotImplementedError

    @abstractmethod
    def posterior_state_dist(
        self: ModelT,
        given_params: ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnosis: dict[str, PatternType] | None = None,
    ) -> np.ndarray:
        """Return the posterior state distribution using the ``given_diagnosis``.

        The posterior state distribution is the probability of the model being in a
        certain state given the diagnosis. The ``given_params`` are passed to the
        :py:meth:`set_params` method. Alternatively to parameters, one may also pass
        a ``given_state_dist``, which is effectively the precomputed prior from calling
        :py:meth:`.state_dist`.
        """
        raise NotImplementedError

    def marginalize(
        self,
        involvement: dict[str, PatternType] | None = None,
        given_state_dist: np.ndarray | None = None,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> float:
        """Marginalize ``given_state_dist`` over matching ``involvement`` patterns.

        Any state that matches the provided ``involvement`` pattern is marginalized
        over. For this, the :py:func:`.matrix.compute_encoding` function is used.

        If ``given_state_dist`` is ``None``, it will be computed by calling
        :py:meth:`.state_dist` with the given ``t_stage`` and ``mode``. These arguments
        are ignored if ``given_state_dist`` is provided.
        """
        raise NotImplementedError

    @abstractmethod
    def risk(
        self,
        involvement: PatternType | None = None,
        given_params: ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnosis: dict[str, PatternType] | None = None,
    ) -> float:
        """Return the risk of ``involvement``, given params/state_dist and diagnosis."""
        raise NotImplementedError
