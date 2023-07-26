"""This module implements the managing descriptor for the diagnostic modalities."""
from __future__ import annotations
import warnings
from typing import Union

import numpy as np

from lymph import models


class Modality:
    """Stores the confusion matrix of a diagnostic modality."""
    def __init__(
        self,
        sensitivity: float,
        specificity: float,
        is_trinary: bool = False,
    ) -> None:
        if not (0. <= sensitivity <= 1. and 0. <= specificity <= 1.):
            raise ValueError("Senstivity and specificity must be between 0 and 1.")

        self.sensitivity = sensitivity
        self.specificity = specificity
        self.is_trinary = is_trinary


    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"sensitivity={self.sensitivity!r}, "
            f"specificity={self.specificity!r}, "
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
            raise ValueError("Confusion matrix must have three rows for trinary models.")

        if not self.is_trinary and value.shape[0] != 2:
            raise ValueError("Confusion matrix must have two rows for binary models.")


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



ModalityDef = Union[Modality, np.ndarray]

class ModalityDict(dict):
    """Dictionary storing every diagnostic `Modality` of a lymph model."""
    def __init__(self, is_trinary: bool = False) -> None:
        self.is_trinary = is_trinary

    def __setitem__(self, key: str, value: ModalityDef, / ) -> None:
        """Set the modality of the lymph model."""
        cls = Clinical

        if type(value) is Modality:
            # we assume the modality to be clinical here, because for a binary model
            # it does not matter, but for a trinary model the base `Modalitiy` class
            # would not work.
            if self.is_trinary:
                warnings.warn("Assuming modality to be clinical.")
            value = cls(value.sensitivity, value.specificity, self.is_trinary)

        elif isinstance(value, Modality):
            # in this case, the user has provided a `Clinical` or `Pathological`
            # modality, so we can just use it after passing the model's type (binary
            # or trinary).
            value.is_trinary = self.is_trinary

        elif isinstance(value, np.ndarray):
            # this should allow users to pass some custom confusion matrix directly.
            # we do check if the matrix is valid, but the `Modalitiy` class may
            # misbehave, e.g. when a recomputation of the confusion matrix is triggered.
            specificity = value[0, 0]
            sensitivity = value[-1, -1]
            modality = Modality(sensitivity, specificity, self.is_trinary)
            modality.confusion_matrix = value
            value = modality

        else:
            # lastly, the user may have provided a list or tuple with the specificity
            # and sensitivity and we're trying to interpret it that way. As before, we
            # assume the modality to be clinical here.
            try:
                specificity, sensitivity = value
                if self.is_trinary:
                    warnings.warn("Assuming modality to be clinical.")
                value = cls(sensitivity, specificity, self.is_trinary)
            except ValueError as val_err:
                raise ValueError(
                    "Value must be a `Clinical` or `Pathological` modality, a "
                    "confusion matrix or a list/tiple with specificity and sensitivity."
                ) from val_err

        super().__setitem__(key, value)


    def update(self, value: dict) -> None:
        for key, value in value.items():
            self[key] = value


class Lookup:
    """Descriptor class for the diagnostic modalities.

    This class is used to manage the diagnostic modalities of a lymph model. It
    provides a descriptor to access the modality of a lymph model. When first trying
    to access this descriptor, it will compute the modality of the model.

    Attributes:
        modality (Modality): modality of the lymph model
    """
    def __set_name__(self, owner, name: str):
        self.private_name = '_' + name


    def __get__(self, instance: models.Unilateral, _cls) -> ModalityDict:
        """Return the modality of the lymph model."""
        if not hasattr(instance, self.private_name):
            modality = ModalityDict(is_trinary=instance.is_trinary)
            setattr(instance, self.private_name, modality)

        return getattr(instance, self.private_name)


    def __set__(self, instance: models.Unilateral, value: Union[ModalityDict, dict]):
        """Set the modality of the lymph model."""
        if not hasattr(instance, self.private_name):
            modality = ModalityDict(is_trinary=instance.is_trinary)
            setattr(instance, self.private_name, modality)

        getattr(instance, self.private_name).update(value)


    def __delete__(self, instance: models.Unilateral):
        """Delete the modality of the lymph model."""
        if hasattr(instance, self.private_name):
            delattr(instance, self.private_name)
