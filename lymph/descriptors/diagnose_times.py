"""
Module that defines helper classes for marginalizing over diagnose times in the
model classes.
"""
from __future__ import annotations

import warnings

import numpy as np

from lymph import models


class SupportError(Exception):
    """Error that is raised when no support for a distribution is provided."""


class Distribution:
    """Class that provides methods for marginalizing over diagnose times."""
    def __init__(
        self,
        distribution: list[float] | np.ndarray | callable,
        max_time: int | None = None,
    ) -> None:
        """Initialize a distribution over diagnose times.

        This object can either be created by passing a parametrized function (e.g.,
        a frozen scipy distribution) or by passing a list of probabilities for each
        diagnose time.
        """
        if max_time is None:
            if callable(distribution):
                raise ValueError("The maximum time must be provided for a callable.")
            max_time = len(distribution) - 1

        if max_time < 0:
            raise ValueError("Maximum time must be positive.")

        if max_time is not None:
            if callable(distribution):
                self._func = distribution
            elif len(distribution) != max_time + 1:
                raise ValueError("Length of distribution and max_time + 1 don't match.")
            else:
                self._frozen = self.normalize(distribution)

        self.support = np.arange(max_time + 1)


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


    def get_param(self) -> float | str:
        """If the distribution is updateable, return the current parameter."""
        if self.is_updateable:
            if hasattr(self, '_param'):
                return self._param
            raise ValueError("Updateable distribution's parameter has not been set.")

        raise ValueError("Distribution is not updateable.")


    def set_param(self, param: float) -> None:
        """Update distribution by setting its parameter and storing the frozen PMF."""
        if self.is_updateable:
            self._param = param
            self._frozen = self.normalize(self._func(self.support, param))
        else:
            warnings.warn("Distribution is not updateable, skipping...")


    def draw(self) -> np.ndarray:
        """Draw sample of diagnose times from the PMF."""
        return np.random.choice(a=self.support, p=self.distribution)


class DistributionDict(dict):
    """
    Class that replicates the behaviour of a dictionary of marginalizors for each
    T-stage, ensuring they all have the same support.
    """
    def __init__(self, *args, max_time: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        if max_time is not None and max_time < 0:
            raise ValueError("Maximum time must be positive.")

        self.max_time = max_time


    def __setitem__(
        self,
        t_stage: str,
        distribution: list[float] | np.ndarray | callable,
    ) -> None:
        """Set the distribution to marginalize over diagnose times for a T-stage."""
        distribution = Distribution(distribution, max_time=self.max_time)
        super().__setitem__(t_stage, distribution)


    def update(self, new_dict: DistributionDict) -> None:
        for t_stage, distribution in new_dict.items():
            self[t_stage] = distribution


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


class DistributionLookup:
    """Descriptor to access the distributions over diagnose times per T-category."""
    def __set_name__(self, owner, name):
        self.private_name = '_' + name


    def __get__(self, instance: models.Unilateral, _cls) -> DistributionDict:
        if not hasattr(instance, self.private_name):
            distribution_dict = DistributionDict(max_time=instance.max_t)
            setattr(instance, self.private_name, distribution_dict)

        return getattr(instance, self.private_name)


    def __set__(self, instance: models.Unilateral, value: DistributionDict):
        self.__delete__(instance)
        self.__get__(instance, type(instance)).update(value)


    def __delete__(self, instance: models.Unilateral):
        """Delete the modality of the lymph model."""
        if hasattr(instance, self.private_name):
            delattr(instance, self.private_name)
