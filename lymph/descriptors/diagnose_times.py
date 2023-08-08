"""
Module that defines helper classes for marginalizing over diagnose times in the
model classes.
"""
from __future__ import annotations

import warnings

import numpy as np

from lymph import models
from lymph.descriptors import AbstractDictDescriptor, AbstractLookupDict


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
        `scipy.stats` distribution) or by passing a list of probabilities for each
        diagnose time.

        The signature of the function must be `func(support, *args, **kwargs)`, where
        `support` is the support of the distribution from 0 to `max_time`. The
        function must return a list of probabilities for each diagnose time.

        Since `max_time` specifies the support of the distribution (rangin from 0 to
        `max_time`), it must be provided if a parametrized function is passed. If a
        list of probabilities is passed, `max_time` is inferred from the length of the
        list and can be omitted. But an error is raised if the length of the list and
        `max_time` + 1 don't match, in case it is accidentally provided.
        """
        if max_time is None:
            if callable(distribution):
                raise ValueError("The maximum time must be provided for a callable.")
            max_time = len(distribution) - 1

        if max_time < 0:
            raise ValueError("Maximum time must be positive.")

        if callable(distribution):
            self._func = distribution
        elif len(distribution) != max_time + 1:
            raise ValueError("Length of distribution and max_time + 1 don't match.")
        else:
            self._frozen = self.normalize(distribution)

        self.support = np.arange(max_time + 1)
        self._args = ()
        self._kwargs = {}


    @classmethod
    def from_instance(cls, other: Distribution) -> Distribution:
        """Create a new distribution from an existing one."""
        if other.is_updateable:
            new_instance = cls(other._func, max_time=other.support[-1])
        else:
            new_instance = cls(other.distribution, max_time=other.support[-1])

        new_instance._args = other._args
        new_instance._kwargs = other._kwargs

        return new_instance


    @staticmethod
    def normalize(distribution: np.ndarray) -> np.ndarray:
        """Normalize a distribution."""
        distribution = np.array(distribution)
        return distribution / np.sum(distribution)


    @property
    def distribution(self) -> np.ndarray:
        """Return the probability mass function of the distribution if it is frozen."""
        if self._frozen is None:
            raise ValueError("Distribution has not been frozen yet")
        return self._frozen


    @property
    def is_frozen(self) -> bool:
        """Return ``True`` if the distribution is frozen."""
        return hasattr(self, '_frozen')


    @property
    def is_updateable(self) -> bool:
        """Return ``True`` if the distribution can be updated by calling `set_param`."""
        return hasattr(self, '_func')


    def get_params(self) -> tuple[tuple, dict]:
        """If the distribution is updateable, return the current parameters."""
        if not self.is_updateable:
            raise ValueError("Distribution is not updateable.")

        return self._args, self._kwargs


    def set_params(self, *args, **kwargs) -> None:
        """Update distribution by setting its parameters and storing the frozen PMF."""
        if self.is_updateable:
            self._args = args
            self._kwargs = kwargs

            try:
                self._frozen = self.normalize(
                    self._func(self.support, *self._args, **self._kwargs)
                )
            except Exception as exc:
                raise ValueError("Error while freezing distribution.") from exc
        else:
            warnings.warn("Distribution is not updateable, skipping...")


    def draw(self) -> np.ndarray:
        """Draw sample of diagnose times from the PMF."""
        return np.random.choice(a=self.support, p=self.distribution)


class DistributionsUserDict(AbstractLookupDict):
    """Dictionary with added methods for storing distributions over diagnose times."""
    # pylint: disable=no-member
    def __setitem__(
        self,
        t_stage: str,
        distribution: list[float] | np.ndarray | Distribution,
    ) -> None:
        """Set the distribution to marginalize over diagnose times for a T-stage."""
        if isinstance(distribution, Distribution):
            distribution = Distribution.from_instance(distribution)
        else:
            distribution = Distribution(distribution, max_time=self.max_time)

        super().__setitem__(t_stage, distribution)


    @property
    def num_parametric(self) -> int:
        """Return the number of parametrized distributions."""
        return sum(distribution.is_updateable for distribution in self.values())


    def set_distribution_params(self, params: list[float] | np.ndarray) -> None:
        """
        Update all marginalizors stored in this instance that are updateable with
        values from the ``params`` argument.

        Use ``stop_quietly`` to avoid raising an error when too few parameters are
        provided to update all distributions.
        """
        if len(params) < self.num_parametric:
            warnings.warn("Not enough parameters to update all distributions.")
        elif len(params) > self.num_parametric:
            warnings.warn("More parameters than distributions to update.")

        for param, distribution in zip(params, self.values()):
            if distribution.is_updateable:
                distribution.set_param(param)


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


class Distributions(AbstractDictDescriptor):
    """Descriptor that adds a dictionary for storing distributions over diagnose times."""
    def _get_callback(self, instance: models.Unilateral):
        """Initialize the lookup dictionary."""
        distribution_dict = DistributionsUserDict(max_time=instance.max_time)
        setattr(instance, self.private_name, distribution_dict)
