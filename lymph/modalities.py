"""
Module implementing management of the diagnostic modalities.

This allows the user to define diagnostic modalities and their sensitivity/specificity
values. This is necessary to compute the likelihood of a dataset (that was created by
recoding the output of diagnostic modalities), given the model and its parameters
(which we want to learn).
"""
from __future__ import annotations

import warnings
from typing import List, Tuple, Union

import numpy as np

from lymph.helper import AbstractLookupDict, trigger


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



ModalityDef = Union[Modality, np.ndarray, Tuple[float, float], List[float]]

class ModalitiesUserDict(AbstractLookupDict):
    """Dictionary storing instances of :py:class:`Modality` for a lymph model.

    This class allows the user to specify the diagnostic modalities of a lymph model
    in a convenient way. The user may pass an instance of :py:class:`Modality` - or one
    of its subclasses - directly. Especially for trinary models, it is recommended to
    use the subclasses :py:class:`Clinical` and :py:class:`Pathological` to avoid
    ambiguities.

    Alternatively, a simple tuple or list of floats may be passed, from which the first
    two entries are interpreted as the specificity and sensitivity, respectively. For
    trinary models, we assume the modality to be :py:class:`Clinical`.

    For completely custom confusion matrices, the user may pass a numpy array directly.
    In the binary case, a valid :py:class:`Modality` instance is constructed from the
    array. For trinary models, the array must have three rows, and is not possible
    anymore to infer the type of the modality or unambiguouse values for sensitivity and
    specificity. This may lead to unexpected results when the confusion matrix is
    recomputed accidentally at some point.

    Examples:

    >>> binary_modalities = ModalitiesUserDict(is_trinary=False)
    >>> binary_modalities["test"] = Modality(0.9, 0.8)
    >>> binary_modalities["test"].confusion_matrix
    array([[0.9, 0.1],
           [0.2, 0.8]])
    >>> modalities = ModalitiesUserDict(is_trinary=True)
    >>> modalities["CT"] = Clinical(specificity=0.9, sensitivity=0.8)
    >>> modalities["CT"].confusion_matrix
    array([[0.9, 0.1],
           [0.9, 0.1],
           [0.2, 0.8]])
    >>> modalities["PET"] = (0.85, 0.82)
    >>> modalities["PET"]
    Clinical(specificity=0.85, sensitivity=0.82, is_trinary=True)
    >>> modalities["pathology"] = Pathological(specificity=1.0, sensitivity=1.0)
    >>> modalities["pathology"].confusion_matrix
    array([[1., 0.],
           [0., 1.],
           [0., 1.]])
    """
    @trigger
    def __setitem__(self, name: str, value: ModalityDef, / ) -> None:
        """Set the modality of the lymph model."""
        # pylint: disable=unidiomatic-typecheck
        # pylint: disable=no-member
        cls = Clinical

        if type(value) is Modality:
            # we assume the modality to be clinical here, because for a binary model
            # it does not matter, but for a trinary model the base `Modalitiy` class
            # would not work.
            if self.is_trinary:
                warnings.warn(f"Assuming modality to be `{cls.__name__}`.")
            value = cls(value.specificity, value.sensitivity, self.is_trinary)

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
            modality = Modality(specificity, sensitivity, self.is_trinary)
            modality.confusion_matrix = value

            if self.is_trinary:
                warnings.warn(
                    "Provided transition matrix will be used as is. The sensitivity "
                    "and specificity extracted from it may be nonsensical. Recomputing "
                    "the confusion matrix from them may not work."
                )

            value = modality

        else:
            # lastly, the user may have provided a list or tuple with the specificity
            # and sensitivity and we're trying to interpret it that way. As before, we
            # assume the modality to be clinical here.
            try:
                specificity, sensitivity = value
                if self.is_trinary:
                    warnings.warn(f"Assuming modality to be `{cls.__name__}`.")
                value = cls(specificity, sensitivity, self.is_trinary)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    "Value must be a `Clinical` or `Pathological` modality, a "
                    "confusion matrix or a list/tuple containing specificity and "
                    "sensitivity."
                ) from err

        super().__setitem__(name, value)


    @trigger
    def __delitem__(self, key: str) -> None:
        return super().__delitem__(key)


    def confusion_matrices_hash(self) -> int:
        """Compute a kind of hash from all confusion matrices.

        Note:
            This is used to check if some modalities have changed and the observation
            matrix needs to be recomputed. It should not be used as a replacement for
            the ``__hash__`` method, for two reasons:

            1. It may change over the lifetime of the object, whereas ``__hash__``
                should be constant.
            2. It only takes into account the ``confusion_matric`` of the modality,
                nothing else.
        """
        confusion_mat_bytes = b""
        for modality in self.values():
            confusion_mat_bytes += modality.confusion_matrix.tobytes()

        return hash(confusion_mat_bytes)
