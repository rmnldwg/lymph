"""
Type aliases and protocols used in the lymph package.
"""
from abc import ABC, abstractmethod
from typing import Iterable, Protocol, TypeVar

import pandas as pd
from pandas._libs.missing import NAType


class HasSetParams(Protocol):
    """Protocol for classes that have a ``set_params`` method."""
    def set_params(self, *args: float, **kwargs: float) -> tuple[float]:
        ...


class HasGetParams(Protocol):
    """Protocol for classes that have a ``get_params`` method."""
    def get_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> tuple[float] | dict[str, float]:
        ...


PatternType = dict[str, bool | NAType | None]
"""Type alias for an involvement pattern."""

DiagnoseType = dict[str, PatternType]
"""Type alias for a diagnose, which is a involvement pattern per diagnostic modality."""


M = TypeVar("M", bound="Model")

class Model(ABC):
    """Abstract base class for models.

    This class provides a scaffold for the methods that any model for lymphatic
    tumor progression should implement.
    """
    @abstractmethod
    def get_params(
        self: M,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> Iterable[float] | dict[str, float]:
        """Return the parameters of the model.

        The parameters are returned as a dictionary if ``as_dict`` is True, and as
        an iterable of floats otherwise. The argument ``as_flat`` determines whether
        the returned dict is flat or nested. This is helpful, because a model may call
        the ``get_params`` method of other instances, which can be fused to get a
        flat dictionary.
        """

    @abstractmethod
    def set_params(self: M, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model.

        The parameters may be passed as positional or keyword arguments. The positional
        arguments are used up one by one by the ``set_params`` methods the model calls.
        Keyword arguments override the positional arguments.
        """

    @abstractmethod
    def load_patient_data(
        self: M,
        patient_data: pd.DataFrame,
    ) -> None:
        """Load patient data in `LyProX`_ format into the model.

        .. _LyProX: https://lyprox.org/
        """

    @abstractmethod
    def likelihood(
        self: M,
        given_param_args: Iterable[float],
        given_param_kwargs: dict[str, float],
        log: bool = True,
    ) -> float:
        """Return the likelihood of the model given the parameters.

        The likelihood is returned in log space if ``log`` is True, and in linear space
        otherwise. The parameters may be passed as positional or keyword arguments.
        They are then passed to the :py:meth:`set_params` method first.
        """
