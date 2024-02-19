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
from typing import Iterable

import numpy as np

from lymph.helper import AbstractLookupDict, flatten, popfirst, set_params_for

logger = logging.getLogger(__name__)


class SupportError(Exception):
    """Error that is raised when no support for a distribution is provided."""


class Distribution:
    """Class that provides a way of storeing distributions over diagnose times."""
    def __init__(
        self,
        distribution: list[float] | np.ndarray | callable,
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

        Since ``max_time`` specifies the support of the distribution (rangin from 0 to
        ``max_time``), it must be provided if a parametrized function is passed. If a
        list of probabilities is passed, ``max_time`` is inferred from the length of the
        list and can be omitted. But an error is raised if the length of the list and
        ``max_time`` + 1 don't match, in case it is accidentally provided.
        """
        self._kwargs = {}

        if callable(distribution):
            if max_time is None:
                raise ValueError("max_time must be provided if a function is passed")
            if max_time < 0:
                raise ValueError("max_time must be a positive integer")

            self.check_callable(distribution)
            self.support = np.arange(max_time + 1)
            self._func = distribution
            self._frozen = self.distribution

        else:
            max_time = self.check_frozen(distribution, max_time)
            self.support = np.arange(max_time + 1)
            self._func = None
            self._frozen = self.normalize(distribution)


    def copy(self) -> Distribution:
        """Return a copy of the distribution.

        Note:
            This will return a frozen distribution, even if the original distribution
            was parametrized.
        """
        return type(self)(
            distribution=self.distribution,
            max_time=self.support[-1],
        )


    @staticmethod
    def check_frozen(distribution: list[float] | np.ndarray, max_time: int) -> int:
        """Check if the frozen distribution is valid.

        The frozen distribution must be a list or array of probabilities for each
        diagnose time. The length of the list must be ``max_time`` + 1.
        """
        if max_time is None:
            max_time = len(distribution) - 1
        elif max_time != len(distribution) - 1:
            raise ValueError("max_time and the length of the distribution don't match")

        return max_time


    def check_callable(self, distribution: callable) -> None:
        """Check if the callable's signature is valid.

        The signature of the provided parametric distribution must be
        ``func(support, **kwargs)``. The first argument is the support of the
        distribution, which is a list or array of integers from 0 to ``max_time``.
        The ``**kwargs`` are keyword parameters that are passed to the function to
        update it.
        """
        # skip the first parameter, which is the support
        skip_first = True
        for name, param in inspect.signature(distribution).parameters.items():
            if skip_first:
                skip_first = False
                continue

            if param.default is inspect.Parameter.empty:
                raise ValueError("All params of the function must be keyword arguments")

            self._kwargs[name] = param.default


    @classmethod
    def from_instance(cls, other: Distribution, max_time: int) -> Distribution:
        """Create a new distribution from an existing one."""
        if other.support[-1] != max_time:
            warnings.warn(
                "max_time of the new distribution is different from the old one. "
                "Support will be truncated/expanded."
            )

        if other.is_updateable:
            new_instance = cls(other._func, max_time=max_time)
            new_instance._kwargs = other._kwargs
        else:
            new_instance = cls(other.distribution[:max_time + 1], max_time=max_time)

        return new_instance


    @staticmethod
    def normalize(distribution: np.ndarray) -> np.ndarray:
        """Normalize a distribution."""
        distribution = np.array(distribution)
        return distribution / np.sum(distribution)


    @property
    def distribution(self) -> np.ndarray:
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

        Parameters can be set via positional arguments or keyword arguments. Keyword
        arguments override positional arguments. If the distribution is not updateable,
        a warning is issued and all args and kwargs are returned.

        If any of the parameters is invalid, a ``ValueError`` is raised and the original
        parameters are restored.

        Unused args and kwargs are returned as well.
        """
        if not self.is_updateable:
            warnings.warn("Distribution is not updateable, ignoring parameters")
            return args

        old_kwargs = self._kwargs.copy()

        for name, value in self._kwargs.items():
            first, args = popfirst(args)
            self._kwargs[name] = first or kwargs.get(name, value)

        try:
            _ = self.distribution
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

        return rng.choice(a=self.support, p=self.distribution, size=num)


class DistributionsUserDict(AbstractLookupDict):
    """Dictionary with added methods for storing distributions over diagnose times."""
    max_time: int

    def __setitem__(
        self,
        t_stage: str,
        distribution: list[float] | np.ndarray | Distribution,
    ) -> None:
        """Set the distribution to marginalize over diagnose times for a T-stage."""
        if isinstance(distribution, Distribution):
            distribution = Distribution.from_instance(distribution, max_time=self.max_time)
        else:
            distribution = Distribution(distribution, max_time=self.max_time)

        super().__setitem__(t_stage, distribution)


    def __delitem__(self, t_stage: str) -> None:
        """Delete the distribution for a T-stage."""
        super().__delitem__(t_stage)


    @property
    def num_parametric(self) -> int:
        """Return the number of parametrized distributions."""
        return sum(distribution.is_updateable for distribution in self.values())


    def get_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> float | Iterable[float] | dict[str, float]:
        """Return the parameters of parametrized distributions.

        If ``as_dict`` is ``False``, return an iterable of all parameter values. If
        ``as_dict`` is ``True``, return a nested dictionary with the T-stages as keys
        and the distributions' parameter dicts as values (essentially what is returned
        by :py:meth:`~lymph.diagnose_times.Distribution.get_params`).

        If ``as_flat`` is ``True``, return a flat dictionary with the T-stages and
        parameters as keys and values, respectively. This is the result of passing the
        nested dictionary to :py:meth:`~lymph.helper.flatten`.
        """
        params = {}

        for t_stage, distribution in self.items():
            if not distribution.is_updateable:
                continue

            params[t_stage] = distribution.get_params(as_flat=as_flat)

        if as_flat or not as_dict:
            params = flatten(params)

        return params if as_dict else params.values()


    def set_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Update all parametrized distributions.

        When the new parameters are provided as positional arguments, they are used up
        in the order of the T-stages.

        If the params are provided as keyword arguments, the keys must be of the form
        ``{t_stage}_{param}``, where ``t_stage`` is the T-stage and ``param`` is the
        name of the parameter to update. Keyword arguments override positional ones.
        """
        return set_params_for(self, *args, **kwargs)


    def draw(
        self,
        prob_of_t_stage: dict[str, float],
        size: int = 1,
    ) -> tuple[list[str], list[int]]:
        """
        Draw first a T-stage and then from that distribution a diagnose time.

        Args:
            dist: Distribution over T-stages. For each key, this defines the
                probability for seeing the respective T-stage. Will be normalized if
                it isn't already.
        """
        stage_dist = np.zeros(shape=len(self))
        t_stages = list(self.keys())

        for i, t_stage in enumerate(t_stages):
            stage_dist[i] = prob_of_t_stage[t_stage]

        stage_dist = stage_dist / np.sum(stage_dist)
        drawn_t_stages = np.random.choice(a=t_stages, p=stage_dist, size=size).tolist()
        drawn_diag_times = [self[t].draw() for t in drawn_t_stages]

        return drawn_t_stages, drawn_diag_times
