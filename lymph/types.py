"""
Type aliases and protocols used in the lymph package.
"""
from abc import ABC, abstractmethod
from typing import Iterable, Literal, Protocol, TypeVar

import numpy as np
import pandas as pd
from pandas._libs.missing import NAType


class DataWarning(UserWarning):
    """Parent class of all warnings related to potential data issues."""


class MissingLNLWarning(DataWarning):
    """Raised when a lymph node level is missing from the data."""


class MissingLateralisationWarning(DataWarning):
    """Raised when a lateralisation is missing from the data."""


class MissingTStageWarning(DataWarning):
    """Warning that is raised when a defined T-stage is missing from the data."""


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


PatternType = dict[str, bool | str | NAType | None]
"""Type alias for an involvement pattern.

An involvement pattern is a dictionary with keys for the lymph node levels and values
for the involvement of the respective lymph nodes. The values are either True, False,
or None, which means that the involvement is unknown.

TODO: Document the new possibilities to specify trinary involvment.
See :py:func:`.matrix.compute_encoding`

>>> pattern = {"I": True, "II": False, "III": None}
"""

DiagnoseType = dict[str, PatternType]
"""Type alias for a diagnose, which is an involvement pattern per diagnostic modality.

>>> diagnose = {
...     "CT": {"I": True, "II": False, "III": None},
...     "MRI": {"I": True, "II": True, "III": None},
... }
"""


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
        raise NotImplementedError

    def get_num_dims(self: M, mode: Literal["HMM", "BN"] = "HMM") -> int:
        """Return the number of dimensions of the parameter space.

        A hidden Markov model (``mode="HMM"``) typically has more parameters than a
        Bayesian network (``mode="BN"``), because it we need parameters for the
        distributions over diagnosis times. Your can read more about that in the
        :py:mod:`lymph.diagnose_times` module.
        """
        # pylint: disable=no-member
        num = len(self.get_params())
        if mode == "BN":
            num -= len(self.get_distribution_params())
        return num

    @abstractmethod
    def set_params(self: M, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model.

        The parameters may be passed as positional or keyword arguments. The positional
        arguments are used up one by one by the ``set_params`` methods the model calls.
        Keyword arguments override the positional arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def load_patient_data(
        self: M,
        patient_data: pd.DataFrame,
    ) -> None:
        """Load patient data in `LyProX`_ format into the model.

        .. _LyProX: https://lyprox.org/
        """
        raise NotImplementedError

    @abstractmethod
    def likelihood(
        self: M,
        given_params: Iterable[float] | dict[str, float] | None = None,
        log: bool = True,
    ) -> float:
        """Return the likelihood of the model given the parameters.

        The likelihood is returned in log space if ``log`` is True, and in linear space
        otherwise. The parameters may be passed as positional or keyword arguments.
        They are then passed to the :py:meth:`set_params` method first.
        """
        raise NotImplementedError

    @abstractmethod
    def risk(
        self,
        involvement: PatternType | None = None,
        given_params: Iterable[float] | dict[str, float] | None = None,
        given_diagnoses: dict[str, PatternType] | None = None,
    ) -> float | np.ndarray:
        """Return the risk of ``involvement``, given the parameters and diagnoses."""
        raise NotImplementedError
