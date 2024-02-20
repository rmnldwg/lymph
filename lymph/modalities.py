"""
Module implementing management of the diagnostic modalities.

This allows the user to define diagnostic modalities and their sensitivity/specificity
values. This is necessary to compute the likelihood of a dataset (that was created by
recoding the output of diagnostic modalities), given the model and its parameters
(which we want to learn).
"""
from __future__ import annotations

import warnings
from abc import ABC
from typing import Literal, TypeVar

import numpy as np


class Modality:
    """Stores the confusion matrix of a diagnostic modality."""
    def __init__(
        self,
        specificity: float,
        sensitivity: float,
        is_trinary: bool = False,
    ) -> None:
        if not (0. <= sensitivity <= 1. and 0. <= specificity <= 1.):
            raise ValueError("Senstivity and specificity must be between 0 and 1.")

        self.specificity = specificity
        self.sensitivity = sensitivity
        self.is_trinary = is_trinary


    def __hash__(self) -> int:
        return hash((self.specificity, self.sensitivity, self.is_trinary))


    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"specificity={self.specificity!r}, "
            f"sensitivity={self.sensitivity!r}, "
            f"is_trinary={self.is_trinary!r})"
        )


    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute the confusion matrix of the modality."""
        return np.array([
            [self.specificity, 1. - self.specificity],
            [1. - self.sensitivity, self.sensitivity],
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
        is_trinary: bool = False,
        modality_children: dict[str, Composite] = None,
        is_modality_leaf: bool = False,
    ) -> None:
        """Initialize the modality composite."""
        self._is_trinary = is_trinary

        if modality_children is None:
            modality_children = {}

        if is_modality_leaf:
            self._modalities = {}
            self._modality_children = {}   # ignore any provided children

        self._modality_children = modality_children
        super().__init__()


    @property
    def _is_modality_leaf(self: MC) -> bool:
        """Return whether the composite is a leaf node."""
        if len(self._modality_children) > 0:
            return False

        if not hasattr(self, "_modalities"):
            raise AttributeError(f"{self} has no children and no modalities.")

        return True


    @property
    def is_trinary(self: MC) -> bool:
        """Return whether the modality is trinary."""
        return self._is_trinary


    def get_modality(self: MC, name: str) -> Modality:
        """Return the modality with the given name."""
        return self.get_all_modalities()[name]


    def get_all_modalities(self: MC) -> dict[str, Modality]:
        """Return all modalities of the composite."""
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
        specificity: float,
        sensitivity: float,
        kind: Literal["clinical", "pathological"] = "clinical",
    ) -> None:
        """Set the modality with the given name."""
        if self._is_modality_leaf:
            cls = Pathological if kind == "pathological" else Clinical
            self._modalities[name] = cls(specificity, sensitivity, self.is_trinary)

        else:
            for child in self._modality_children.values():
                child.set_modality(name, specificity, sensitivity, kind)


    def replace_all_modalities(self: MC, modalities: dict[str, Modality]) -> None:
        """Replace all modalities of the composite."""
        if self._is_modality_leaf:
            self._modalities = {}
            for name, modality in modalities.items():
                kind = "pathological" if isinstance(modality, Pathological) else "clinical"
                self.set_modality(name, modality.specificity, modality.sensitivity, kind)

        else:
            for child in self._modality_children.values():
                child.replace_all_modalities(modalities)


    def compute_modalities_hash(self: MC) -> int:
        """Compute a hash from all modalities."""
        hash_res = 0
        if self._is_modality_leaf:
            for name, modality in self._modalities.items():
                hash_res = hash((hash_res, name, hash(modality)))

        for child in self._modality_children.values():
            hash_res = hash((hash_res, hash(child)))

        return hash_res
