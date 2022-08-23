"""
Module that defines a helper class for marginalizing over diagnose times in the
model classes.
"""
import warnings
from typing import Callable, List, Optional, Union

import numpy as np


class Marginalizor:
    """
    Class that provides methods for marginalizing over diagnose times.
    """
    def __init__(
        self,
        dist: Optional[Union[List[float], np.ndarray]] = None,
        func: Optional[Callable] = None,
        max_t: int = 10,
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
        self.support = np.arange(max_t + 1)
        if dist is not None:
            self.pmf = dist
            self._func = None
        elif func is not None:
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
    def pmf(self, dist: Union[List[float], np.ndarray]) -> None:
        """
        Set the probability mass function of the marginalizor and make sure
        it is normalized.
        """
        dist_arr = np.array(dist)
        cum_dist = np.sum(dist_arr)
        if not np.isclose(cum_dist, 1.):
            warnings.warn(
                "Probability mass function does not sum to 1, will be normalized."
            )
        if not dist_arr.shape == self.support.shape:
            raise ValueError(f"Distribution must be of shape {self.support.shape}")
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

    def update(self, param: float) -> None:
        """
        Update the marginalizor by providing the function with its parameter and
        storing the resulting PMF.
        """
        if self.is_updateable:
            self.pmf = self._func(self.support, param)
        else:
            raise RuntimeError("Marginalizor is not updateable.")


class MarginalizorDict(dict):
    """
    Class that stores a dictionary of marginalizors for each T-stage and ensures
    they all have the same support.
    """
    def __init__(self, *args, max_t: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_t = max_t

    def __setitem__(
        self,
        t_stage: str,
        dist: Union[List[float], np.ndarray, Callable]
    ) -> None:
        """
        Define the (frozen or unfrozen) marginalizor for a given T-stage. If the
        provided dist is listlike, the PMF will be frozen. If it is callable, the
        PMF will only be frozen after the instance of the Marginalizor is called with
        the parameters of the function (except the first one, which is the support).
        """
        if callable(dist):
            marg = Marginalizor(func=dist, max_t=self.max_t)
        else:
            marg = Marginalizor(dist=dist, max_t=self.max_t)
        super().__setitem__(t_stage, marg)

    def update(self, params: Union[List[float], np.ndarray]) -> None:
        """
        Update all marginalizors stored in this instance that are updateable.
        """
        params_iterator = iter(params)
        for _, marg in self.items():
            if marg.is_updateable:
                try:
                    marg.update(next(params_iterator))
                except StopIteration:
                    raise ValueError("Not enough parameters provided")
