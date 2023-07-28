"""
Module that defines helper classes for marginalizing over diagnose times in the
model classes.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from lymph import models


class SupportError(Exception):
    """Error that is raised when no support for a distribution is provided."""


class Distribution:
    """Class that provides methods for marginalizing over diagnose times."""
    def __init__(
        self,
        dist: list[float] | np.ndarray | None = None,
        func: Callable | None = None,
        max_t: int | None = None,
    ) -> None:
        """
        Initialize the marginalizor either with a function or with a fixed distribution.

        Args:
            dist: Array or list of probabilities to be used as a frozen distribution.
            func: If this function is provided, initialize distribution as unfrozen,
                meaning that the instance needs to be called with the function's
                parameters to get the PMF for its support and freeze it.
            max_t: Support of the marginalization function runs from 0 to max_t.
        """
        max_t = len(dist) - 1 if max_t is None else max_t
        self.support = np.arange(max_t + 1)

        if dist is not None:
            self.pmf = dist
            self._func = None
        elif func is not None:
            self._param = None
            self._pmf = None
            self._func = func
        else:
            raise ValueError("Either dist or func must be specified")


    @property
    def pmf(self) -> np.ndarray:
        """
        Return the probability mass function of the marginalizor if it is frozen.
        Raise an error otherwise.
        """
        if self._pmf is None:
            raise ValueError("Marginalizor has not been frozen yet")
        return self._pmf

    @pmf.setter
    def pmf(self, dist: list[float] | np.ndarray) -> None:
        """
        Set the probability mass function of the marginalizor and make sure
        it is normalized.
        """
        if callable(dist):
            raise TypeError(
                "Parametrized distribution can only be provided in the constructor. "
                "This method can only set a new frozen distribution."
            )
        dist_arr = np.array(dist)
        cum_dist = np.sum(dist_arr)
        if not np.isclose(cum_dist, 1.):
            warnings.warn(
                "Probability mass function does not sum to 1, will be normalized."
            )
        if not dist_arr.shape == self.support.shape:
            raise ValueError(
                f"Distribution must be of shape {self.support.shape}, "
                f"but has shape {dist_arr.shape}"
            )
        self._pmf = dist_arr / cum_dist

    @property
    def is_frozen(self) -> bool:
        """
        Return True if the marginalizor is frozen.
        """
        return self._pmf is not None

    @property
    def is_updateable(self) -> bool:
        """
        Return True if the marginalizor's PMF can be changed by calling `update`.
        """
        return self._func is not None


    def get_param(self) -> float | str:
        """If the marginalizor is updateable, return the current parameter."""
        if self.is_updateable:
            return self._param or "not set"
        return "not parametrized"

    def update(self, param: float) -> None:
        """
        Update the marginalizor by providing the function with its parameter and
        storing the resulting PMF.
        """
        if self.is_updateable:
            self._param = param
            self.pmf = self._func(self.support, param)
        else:
            warnings.warn("Distribution is not updateable, skipping...")

    def draw(self) -> np.ndarray:
        """Draw sample of diagnose times from the PMF."""
        return np.random.choice(a=self.support, p=self.pmf)


class DistributionDict(dict):
    """
    Class that replicates the behaviour of a dictionary of marginalizors for each
    T-stage, ensuring they all have the same support.
    """
    def __init__(self, *args, max_t: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_t = max_t

    def __setitem__(
        self,
        t_stage: str,
        distribution: list[float] | np.ndarray | Callable
    ) -> None:
        """Set the distribution to marginalize over diagnose times for a T-stage."""
        if self.max_t is None:
            if callable(distribution):
                raise SupportError(
                    "Cannot assign parametric distribution without first defining "
                    "its support"
                )
            self.max_t = len(distribution) - 1

        if callable(distribution):
            marg = Distribution(func=distribution, max_t=self.max_t)
        else:
            marg = Distribution(dist=distribution, max_t=self.max_t)

        super().__setitem__(t_stage, marg)


    def update(self, new_dict: DistributionDict) -> None:
        for t_stage, distribution in new_dict.items():
            self[t_stage] = distribution


    @property
    def num_parametric(self) -> int:
        """Return the number of parametrized distributions."""
        return sum(marg.is_updateable for marg in self.values())


    def update_distributions(self, params: list[float] | np.ndarray) -> None:
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
                distribution.update(param)


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
            distribution_dict = DistributionDict(max_t=instance.max_t)
            setattr(instance, self.private_name, distribution_dict)

        return getattr(instance, self.private_name)


    def __set__(self, instance: models.Unilateral, value: DistributionDict):
        self.__delete__(instance)
        self.__get__(instance, type(instance)).update(value)


    def __delete__(self, instance: models.Unilateral):
        """Delete the modality of the lymph model."""
        if hasattr(instance, self.private_name):
            delattr(instance, self.private_name)
