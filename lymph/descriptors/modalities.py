"""This module implements the managing descriptor for the diagnostic modalities."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from lymph import models


class Modality:
    """Stores the confusion matrix of a diagnostic modality.

    Attributes:
        confusion_matrix (np.ndarray): confusion matrix of modality
    """
    def __init__(
        self,
        sensitivity: Optional[float] = None,
        specificity: Optional[float] = None,
        is_pathological: bool = False,
        confusion_matrix: Optional[np.ndarray] = None,
    ) -> None:
        if sensitivity is not None and specificity is not None:
            pass
        elif confusion_matrix is not None:
            self.confusion_matrix = confusion_matrix
        else:
            raise ValueError(
                "Either confusion matrix or sensitivity and specificity must be given!"
            )


class ModalityDict(dict):
    """Dictionary storing every diagnostic `Modality` of a lymph model."""
    def __setitem__(self, __key: str, __value: Union[Modality, np.ndarray]) -> None:
        if not isinstance(__value, (Modality, np.ndarray)):
            raise TypeError(
                "Modality must be a `Modality` or a confusion matrix (numpy array)!"
            )
        __value = Modality(__value)
        return super().__setitem__(__key, __value)

    def update(self, __value: dict) -> None:
        for key, value in __value.items():
            self.__setitem__(key, value)


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


    def __get__(self, instance: models.Unilateral, _cls) -> Modality:
        """Return the modality of the lymph model."""
        if not hasattr(instance, self.private_name):
            modality = ModalityDict()
            setattr(instance, self.private_name, modality)

        return getattr(instance, self.private_name)


    def __set__(self, instance: models.Unilateral, value: ModalityDict):
        """Set the modality of the lymph model."""
        if not hasattr(instance, self.private_name):
            modality = ModalityDict()
            setattr(instance, self.private_name, modality)

        getattr(instance, self.private_name).update(value)


    def __delete__(self, instance: models.Unilateral):
        """Delete the modality of the lymph model."""
        if hasattr(instance, self.private_name):
            delattr(instance, self.private_name)
