"""
Module implementing management of the diagnostic modalities.

This allows the user to define diagnostic modalities and their sensitivity/specificity
values. This is necessary to compute the likelihood of a dataset (that was created by
recoding the output of diagnostic modalities), given the model and its parameters
(which we want to learn).
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Literal, TypeVar

import numpy as np


class Modality:
    """Stores the confusion matrix of a diagnostic modality."""
    def __init__(
        self,
        spec: float,
        sens: float,
        is_trinary: bool = False,
    ) -> None:
        if not (0. <= sens <= 1. and 0. <= spec <= 1.):
            raise ValueError("Senstivity and specificity must be between 0 and 1.")

        self.spec = spec
        self.sens = sens
        self.is_trinary = is_trinary


    def __hash__(self) -> int:
        """Return a hash of the modality.

        This is computed from the confusion matrix of the modality.
        """
        return hash(self.confusion_matrix.tobytes())


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Modality):
            return False

        return np.all(self.confusion_matrix == other.confusion_matrix)


    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"spec={self.spec!r}, "
            f"sens={self.sens!r}, "
            f"is_trinary={self.is_trinary!r})"
        )


    @property
    def spec(self) -> float:
        """Return the specificity of the modality."""
        return self._spec

    @spec.setter
    def spec(self, value: float) -> None:
        """Set the specificity of the modality."""
        if not 0. <= value <= 1.:
            raise ValueError("Specificity must be between 0 and 1.")

        if hasattr(self, "_confusion_matrix"):
            del self._confusion_matrix

        self._spec = value


    @property
    def sens(self) -> float:
        """Return the sensitivity of the modality."""
        return self._sens

    @sens.setter
    def sens(self, value: float) -> None:
        """Set the sensitivity of the modality."""
        if not 0. <= value <= 1.:
            raise ValueError("Sensitivity must be between 0 and 1.")

        if hasattr(self, "_confusion_matrix"):
            del self._confusion_matrix

        self._sens = value


    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute the confusion matrix of the modality."""
        return np.array([
            [self.spec, 1. - self.spec],
            [1. - self.sens, self.sens],
        ])

    @property
    def confusion_matrix(self) -> np.ndarray:
        """Return the confusion matrix of the modality."""
        if not hasattr(self, '_confusion_matrix'):
            self.confusion_matrix = self.compute_confusion_matrix()

        if self.is_trinary and not self._confusion_matrix.shape[0] == 3:
            self.confusion_matrix = self.compute_confusion_matrix()

        return self._confusion_matrix

    @confusion_matrix.setter
    def confusion_matrix(self, value: np.ndarray) -> None:
        """Set the confusion matrix of the modality."""
        self.check_confusion_matrix(value)
        self._confusion_matrix = value

    def check_confusion_matrix(self, value: np.ndarray) -> None:
        """Check if the confusion matrix is valid."""
        row_sums = np.sum(value, axis=1)
        if not np.allclose(row_sums, 1.):
            raise ValueError("Rows of confusion matrix must sum to one.")

        if not np.all(np.greater_equal(value, 0.)):
            raise ValueError("Confusion matrix must be non-negative.")

        if not np.all(np.less_equal(value, 1.)):
            raise ValueError("Confusion matrix must be less than or equal to one.")

        if self.is_trinary and value.shape[0] != 3:
            raise ValueError("Confusion matrix must have 3 rows for trinary models.")

        if not self.is_trinary and value.shape[0] != 2:
            raise ValueError("Confusion matrix must have 2 rows for binary models.")


class Clinical(Modality):
    """Stores the confusion matrix of a clinical modality."""
    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute the confusion matrix of the clinical modality."""
        binary_confusion_matrix = super().compute_confusion_matrix()
        if not self.is_trinary:
            return binary_confusion_matrix

        return np.vstack([binary_confusion_matrix[0], binary_confusion_matrix])


class Pathological(Modality):
    """Stores the confusion matrix of a pathological modality."""
    def compute_confusion_matrix(self) -> np.ndarray:
        """Return the confusion matrix of the pathological modality."""
        binary_confusion_matrix = super().compute_confusion_matrix()
        if not self.is_trinary:
            return binary_confusion_matrix

        return np.vstack([binary_confusion_matrix, binary_confusion_matrix[1]])



MC = TypeVar("MC", bound="Composite")

class Composite(ABC):
    """Abstract base class implementing the composite pattern for diagnostic modalities.

    Any class inheriting from this class should be able to handle the definition of
    diagnostic modalities and their sensitivity/specificity values,
    """
    _is_trinary: bool
    _modalities: dict[str, Modality]    # only for leaf nodes
    _modality_children: dict[str, Composite]

    def __init__(
        self: MC,
        modality_children: dict[str, Composite] = None,
        is_modality_leaf: bool = False,
    ) -> None:
        """Initialize the modality composite."""
        if modality_children is None:
            modality_children = {}

        if is_modality_leaf:
            self._modalities = {}
            modality_children = {}   # ignore any provided children

        self._modality_children = modality_children


    @property
    def _is_modality_leaf(self: MC) -> bool:
        """Return whether the composite is a leaf node."""
        if len(self._modality_children) > 0:
            return False

        if not hasattr(self, "_modalities"):
            raise AttributeError(f"{self} has no children and no modalities.")

        return True


    @property
    @abstractmethod
    def is_trinary(self: MC) -> bool:
        """Return whether the modality is trinary."""


    def modalities_hash(self: MC) -> int:
        """Compute a hash from all stored modalities.

        See the :py:meth:`.Modality.__hash__` method for more information.
        """
        hash_res = 0
        if self._is_modality_leaf:
            for name, modality in self._modalities.items():
                hash_res = hash((hash_res, name, hash(modality)))

        else:
            for child in self._modality_children.values():
                hash_res = hash((hash_res, child.modalities_hash()))

        return hash_res


    def get_modality(self: MC, name: str) -> Modality:
        """Return the modality with the given ``name``."""
        return self.get_all_modalities()[name]


    def get_all_modalities(self: MC) -> dict[str, Modality]:
        """Return all modalities of the composite.

        This will issue a warning if it finds that not all modalities of the composite
        are equal. Note that it will always return the modalities of the first child.
        This means one should NOT try to set the modalities via the returned dictionary
        of this method. Instead, use the :py:meth:`.set_modality` method.
        """
        if self._is_modality_leaf:
            return self._modalities

        child_keys = list(self._modality_children.keys())
        first_child = self._modality_children[child_keys[0]]
        firs_modalities = first_child.get_all_modalities()
        are_all_equal = True
        for key in child_keys[1:]:
            other_child = self._modality_children[key]
            are_all_equal &= firs_modalities == other_child.get_all_modalities()

        if not are_all_equal:
            warnings.warn("Not all modalities are equal. Returning first one.")

        return firs_modalities


    def set_modality(
        self,
        name: str,
        spec: float,
        sens: float,
        kind: Literal["clinical", "pathological"] = "clinical",
    ) -> None:
        """Set the modality with the given ``name``."""
        if self._is_modality_leaf:
            cls = Pathological if kind == "pathological" else Clinical
            self._modalities[name] = cls(spec, sens, self.is_trinary)

        else:
            for child in self._modality_children.values():
                child.set_modality(name, spec, sens, kind)


    def del_modality(self: MC, name: str) -> None:
        """Delete the modality with the given ``name``."""
        if self._is_modality_leaf:
            del self._modalities[name]

        else:
            for child in self._modality_children.values():
                child.del_modality(name)


    def replace_all_modalities(self: MC, modalities: dict[str, Modality]) -> None:
        """Replace all modalities of the composite with new ``modalities``."""
        if self._is_modality_leaf:
            self.clear_modalities()
            for name, modality in modalities.items():
                kind = "pathological" if isinstance(modality, Pathological) else "clinical"
                self.set_modality(name, modality.spec, modality.sens, kind)

        else:
            for child in self._modality_children.values():
                child.replace_all_modalities(modalities)


    def clear_modalities(self: MC) -> None:
        """Clear all modalities of the composite."""
        if self._is_modality_leaf:
            self._modalities.clear()

        else:
            for child in self._modality_children.values():
                child.clear_modalities()
