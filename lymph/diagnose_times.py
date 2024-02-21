"""
Module for marginalizing over diagnose times.

The hidden Markov model we implement assumes that every patient started off with a
healthy neck, meaning no lymph node levels harboured any metastases. This is a valid
assumption, but brings with it the issue of determining *how long ago* this likely was.

This module allows the user to define a distribution over abstract time-steps that
indicate for different T-categories how probable a diagnosis at this time-step was.
That allows us to treat T1 and T4 patients fundamentally in the same way, even with the
same parameters, except for the parametrization of their respective distribution over
the time of diagnosis.
"""
from __future__ import annotations

import inspect
import logging
import warnings
from abc import ABC
from typing import Any, Iterable, TypeVar

import numpy as np

from lymph.helper import flatten, popfirst, unflatten_and_split

logger = logging.getLogger(__name__)


class SupportError(Exception):
    """Error that is raised when no support for a distribution is provided."""


class Distribution:
    """Class that provides a way of storing distributions over diagnose times."""
    def __init__(
        self,
        distribution: Iterable[float] | callable,
        max_time: int | None = None,
    ) -> None:
        """Initialize a distribution over diagnose times.

        This object can either be created by passing a parametrized function (e.g.,
        ``scipy.stats`` distribution) or by passing a list of probabilities for each
        diagnose time.

        The signature of the function must be ``func(support, **kwargs)``, where
        ``support`` is the support of the distribution from 0 to ``max_time``. The
        function must return a list of probabilities for each diagnose time.

        Note:
            All arguments except ``support`` must have default values and if some
            parameters have bounds (like the binomial distribution's ``p``), the
            function must raise a ``ValueError`` if the parameter is invalid.

        Since ``max_time`` specifies the support of the distribution (ranging from 0 to
        ``max_time``), it must be provided if a parametrized function is passed. If a
        list of probabilities is passed, ``max_time`` is inferred from the length of the
        list and can be omitted. But an error is raised if the length of the list and
        ``max_time`` + 1 don't match, in case it is accidentally provided.
        """
        if callable(distribution):
            self._init_from_callable(distribution, max_time)
        else:
            self._init_from_frozen(distribution, max_time)


    def _init_from_frozen(self, distribution: Iterable[float], max_time: int):
        """Initialize the distribution from a frozen distribution."""
        if max_time is None:
            max_time = len(distribution) - 1

        if max_time != len(distribution) - 1:
            raise ValueError("max_time and the length of the distribution don't match")

        self.support = np.arange(max_time + 1)
        self._kwargs = {}
        self._func = None
        self._frozen = self.normalize(distribution)


    def _init_from_callable(self, distribution: callable, max_time: int | None = None):
        """Initialize the distribution from a callable distribution."""
        if max_time is None:
            raise ValueError("max_time must be provided if a function is passed")
        if max_time < 0:
            raise ValueError("max_time must be a positive integer")

        self.support = np.arange(max_time + 1)
        self._kwargs = self.extract_kwargs(distribution)
        self._func = distribution
        self._frozen = self.pmf


    @staticmethod
    def extract_kwargs(distribution: callable) -> dict[str, Any]:
        """Extract the keyword arguments from the provided parametric distribution.

        The signature of the provided parametric distribution must be
        ``func(support, **kwargs)``. The first argument is the support of the
        distribution, which is a list or array of integers from 0 to ``max_time``.
        The ``**kwargs`` are keyword parameters that are passed to the function to
        update it.
        """
        kwargs = {}
        # skip the first parameter, which is the support
        skip_first = True
        for name, param in inspect.signature(distribution).parameters.items():
            if skip_first:
                skip_first = False
                continue

            if param.default is inspect.Parameter.empty:
                raise ValueError("All params of the function must be keyword arguments")

            kwargs[name] = param.default

        return kwargs


    def __repr__(self) -> str:
        return f"Distribution({repr(self.pmf.tolist())})"


    def __eq__(self, other) -> bool:
        if not isinstance(other, Distribution):
            return False

        return (
            self.is_updateable == other.is_updateable
            and self._kwargs == other._kwargs
            and np.all(self.pmf == other.pmf)
        )


    def __hash__(self) -> int:
        kwarg_tpl = tuple(self._kwargs.items())
        return hash((self.is_updateable, kwarg_tpl, self.pmf.tobytes()))


    @staticmethod
    def normalize(distribution: np.ndarray) -> np.ndarray:
        """Normalize a distribution."""
        distribution = np.array(distribution)
        return distribution / np.sum(distribution)


    @property
    def pmf(self) -> np.ndarray:
        """Return the probability mass function of the distribution if it is frozen."""
        if not hasattr(self, "_frozen") or self._frozen is None:
            self._frozen = self.normalize(self._func(self.support, **self._kwargs))
        return self._frozen


    @property
    def is_updateable(self) -> bool:
        """``True`` if instance can be updated via :py:meth:`~set_param`."""
        return self._func is not None


    def get_params(
        self,
        as_dict: bool = True,
        **_kwargs,
    ) -> float | Iterable[float] | dict[str, float]:
        """If updateable, return the dist's ``param`` value or all params in a dict.

        See Also:
            :py:meth:`lymph.diagnose_times.DistributionsUserDict.get_params`
            :py:meth:`lymph.graph.Edge.get_params`
            :py:meth:`lymph.models.Unilateral.get_params`
            :py:meth:`lymph.models.Bilateral.get_params`
        """
        if not self.is_updateable:
            warnings.warn("Distribution is not updateable, returning empty dict")
            return {} if as_dict else None

        return self._kwargs if as_dict else self._kwargs.values()


    def set_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Update distribution by setting its parameters and storing the frozen PMF.

        Parameters can be set via positional arguments - which are used up one by one
        in the order they are provided and are then returned - or keyword arguments.
        Keyword arguments override positional arguments. If the distribution is not
        updateable, a warning is issued and all args and kwargs are returned.

        If any of the parameters is invalid, a ``ValueError`` is raised and the original
        parameters are restored.
        """
        if not self.is_updateable:
            warnings.warn("Distribution is not updateable, ignoring parameters")
            return args

        old_kwargs = self._kwargs.copy()

        for name, value in self._kwargs.items():
            first, args = popfirst(args)
            self._kwargs[name] = first or kwargs.get(name, value)
            del self._frozen

        try:
            _ = self.pmf
        except ValueError as val_err:
            self._kwargs = old_kwargs
            raise ValueError("Invalid params provided to distribution") from val_err

        return args


    def draw_diag_times(
        self,
        num: int | None = None,
        rng: np.random.Generator | None = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Draw ``num`` samples of diagnose times from the stored PMF.

        A random number generator can be provided as ``rng``. If ``None``, a new one
        is initialized with the given ``seed`` (or ``42``, by default).
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        return rng.choice(a=self.support, p=self.pmf, size=num)



DC = TypeVar("DC", bound="Composite")

class Composite(ABC):
    """Abstract base class implementing the composite pattern for distributions.

    Any class inheriting from this class should be able to handle the definition of
    distributions over diagnosis times.

    Example:

    >>> class MyComposite(Composite):
    ...     pass
    >>> leaf1 = MyComposite(is_distribution_leaf=True)
    >>> leaf2 = MyComposite(is_distribution_leaf=True)
    >>> leaf3 = MyComposite(is_distribution_leaf=True)
    >>> branch1 = MyComposite(distribution_children={"L1": leaf1, "L2": leaf2})
    >>> branch2 = MyComposite(distribution_children={"L3": leaf3})
    >>> root = MyComposite(distribution_children={"B1": branch1, "B2": branch2})
    >>> root.set_distribution("T1", Distribution([0.1, 0.9]))
    >>> root.get_distribution("T1")
    Distribution([0.1 0.9])
    >>> leaf1.get_distribution("T1")
    Distribution([0.1 0.9])
    """
    _max_time: int
    _distributions: dict[str, Distribution]    # only for leaf nodes
    _distribution_children: dict[str, Composite]

    def __init__(
        self: DC,
        max_time: int = 10,
        distribution_children: dict[str, Composite] | None = None,
        is_distribution_leaf: bool = False,
    ) -> None:
        """Initialize the distribution composite."""
        if distribution_children is None:
            distribution_children = {}

        if is_distribution_leaf:
            self._distributions = {}
            self._distribution_children = {}    # ignore any provided children
            self.max_time = max_time            # only set max_time in leaf

        self._distribution_children = distribution_children

        super().__init__()


    @property
    def _is_distribution_leaf(self: DC) -> bool:
        """Return whether the object is a leaf node w.r.t. distributions."""
        if len(self._distribution_children) > 0:
            return False

        if not hasattr(self, "_distributions"):
            raise AttributeError(f"{self} has no children and no distributions.")

        return True


    @property
    def max_time(self: DC) -> int:
        """Return the maximum time for the distributions."""
        if self._is_distribution_leaf:
            return self._max_time

        max_times = {child.max_time for child in self._distribution_children.values()}
        if len(max_times) > 1:
            warnings.warn("Not all max_times are equal. Returning the first one.")

        return list(self._distribution_children.values())[0].max_time

    @max_time.setter
    def max_time(self: DC, value: int) -> None:
        """Set the maximum time for the distributions."""
        if self._is_distribution_leaf:
            if value < 0:
                raise ValueError("max_time must be a positive integer")

            self._max_time = value

        else:
            for child in self._distribution_children.values():
                child.max_time = value


    @property
    def t_stages(self: DC) -> list[str]:
        """Return the T-stages for which distributions are defined."""
        return list(self.get_all_distributions().keys())


    def get_distribution(self: DC, t_stage: str) -> Distribution:
        """Return the distribution for the given ``t_stage``."""
        return self.get_all_distributions()[t_stage]


    def get_all_distributions(self: DC) -> dict[str, Distribution]:
        """Return all distributions.

        This will issue a warning if it finds that not all distributions of the
        composite are equal. Note that it will always return the distributions of the
        first child. This means one should NOT try to set the distributions via the
        returned dictionary of this method. Instead, use the :py:meth:`set_modality`
        method.
        """
        if self._is_distribution_leaf:
            return self._distributions

        child_keys = list(self._distribution_children.keys())
        first_child = self._distribution_children[child_keys[0]]
        first_distributions = first_child.get_all_distributions()
        are_all_equal = True
        for key in child_keys[1:]:
            other_child = self._distribution_children[key]
            are_all_equal &= first_distributions == other_child.get_all_distributions()

        if not are_all_equal:
            warnings.warn("Not all distributions are equal. Returning the first one.")

        return first_distributions


    def set_distribution(
        self: DC,
        t_stage: str,
        distribution: Distribution | Iterable[float] | callable,
    ) -> None:
        """Set/update the distribution for the given ``t_stage``."""
        if self._is_distribution_leaf:
            self._distributions[t_stage] = Distribution(distribution, self.max_time)

        else:
            for child in self._distribution_children.values():
                child.set_distribution(t_stage, distribution)


    def replace_all_distributions(self: DC, distributions: dict[str, Distribution]) -> None:
        """Replace all distributions with the given ones."""
        if self._is_distribution_leaf:
            self._distributions = {}
            for t_stage, distribution in distributions.items():
                self.set_distribution(t_stage, distribution)

        else:
            for child in self._distribution_children.values():
                child.replace_all_distributions(distributions)


    def clear_distributions(self: DC) -> None:
        """Remove all distributions."""
        if self._is_distribution_leaf:
            self._distributions.clear()

        else:
            for child in self._distribution_children.values():
                child.clear_distributions()


    def distributions_hash(self: DC) -> int:
        """Return a hash of all distributions."""
        hash_res = 0
        if self._is_distribution_leaf:
            for t_stage, distribution in self._distributions.items():
                hash_res = hash((hash_res, t_stage, hash(distribution)))

        else:
            for child in self._distribution_children.values():
                hash_res = hash((hash_res, child.distributions_hash()))

        return hash_res


    def get_distribution_params(
        self: DC,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> Iterable[float] | dict[str, float]:
        """Return the parameters of all distributions."""
        params = {}

        if self._is_distribution_leaf:
            for t_stage, distribution in self._distributions.items():
                params[t_stage] = distribution.get_params(as_flat=as_flat)
        else:
            child_keys = list(self._distribution_children.keys())
            first_child = self._distribution_children[child_keys[0]]
            params = first_child.get_distribution_params(as_flat=as_flat)
            are_all_equal = True
            for key in child_keys[1:]:
                other_child = self._distribution_children[key]
                other_params = other_child.get_distribution_params(as_flat=as_flat)
                are_all_equal &= params == other_params

        if as_flat or not as_dict:
            params = flatten(params)

        return params if as_dict else params.values()


    def set_distribution_params(self: DC, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of all distributions."""
        if self._is_distribution_leaf:
            kwargs, global_kwargs = unflatten_and_split(
                kwargs, expected_keys=self._distributions.keys()
            )
            for t_stage, distribution in self._distributions.items():
                t_stage_kwargs = global_kwargs.copy()
                t_stage_kwargs.update(kwargs.get(t_stage, {}))
                args = distribution.set_params(*args, **t_stage_kwargs)

        else:
            kwargs, global_kwargs = unflatten_and_split(
                kwargs, expected_keys=self._distribution_children.keys()
            )
            for key, child in self._distribution_children.items():
                child_kwargs = global_kwargs.copy()
                child_kwargs.update(kwargs.get(key, {}))
                args = child.set_distribution_params(*args, **child_kwargs)

        return args
