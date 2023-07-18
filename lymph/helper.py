"""Module containing supporting classes and functions."""

import warnings
from typing import Optional, List

import numpy as np


class Param:
    """Stores getter and setter functions for a parameter."""
    def __init__(self, getter, setter):
        self.get = getter
        self.set = setter


def check_spsn(spsn: List[float]):
    """Private method that checks whether specificity and sensitvity
    are valid.

    Args:
        spsn (list): list with specificity and sensiticity

    Raises:
        ValueError: raises a value error if the spec or sens is not a number btw. 0.5 and 1.0
    """
    has_len_2 = len(spsn) == 2
    is_above_lb = np.all(np.greater_equal(spsn, 0.5))
    is_below_ub = np.all(np.less_equal(spsn, 1.))
    if not has_len_2 or not is_above_lb or not is_below_ub:
        msg = ("For each modality provide a list of two decimals "
            "between 0.5 and 1.0 as specificity & sensitivity "
            "respectively.")
        raise ValueError(msg)


def change_base(
    number: int,
    base: int,
    reverse: bool = False,
    length: Optional[int] = None
) -> str:
    """Convert an integer into another base.

    Args:
        number: Number to convert
        base: Base of the resulting converted number
        reverse: If true, the converted number will be printed in reverse order.
        length: Length of the returned string. If longer than would be
            necessary, the output will be padded.

    Returns:
        The (padded) string of the converted number.
    """
    if number < 0:
        raise ValueError("Cannot convert negative numbers")
    if base > 16:
        raise ValueError("Base must be 16 or smaller!")
    elif base < 2:
        raise ValueError("There is no unary number system, base must be > 2")

    convert_string = "0123456789ABCDEF"
    result = ''

    if number == 0:
        result += '0'
    else:
        while number >= base:
            result += convert_string[number % base]
            number = number//base
        if number > 0:
            result += convert_string[number]

    if length is None:
        length = len(result)
    elif length < len(result):
        length = len(result)
        warnings.warn("Length cannot be shorter than converted number.")

    pad = '0' * (length - len(result))

    if reverse:
        return result + pad
    else:
        return pad + result[::-1]


def check_modality(modality: str, spsn: list):
    """Private method that checks whether all inserted values
    are valid for a confusion matrix.

    Args:
        modality (str): name of the modality
        spsn (list): list with specificity and sensiticity

    Raises:
        TypeError: returns a type error if the modality is not a string
        ValueError: raises a value error if the spec or sens is not a number btw. 0.5 and 1.0
    """
    if not isinstance(modality, str):
            msg = ("Modality names must be strings.")
            raise TypeError(msg)
    has_len_2 = len(spsn) == 2
    is_above_lb = np.all(np.greater_equal(spsn, 0.5))
    is_below_ub = np.all(np.less_equal(spsn, 1.))
    if not has_len_2 or not is_above_lb or not is_below_ub:
        msg = ("For each modality provide a list of two decimals "
            "between 0.5 and 1.0 as specificity & sensitivity "
            "respectively.")
        raise ValueError(msg)


def clinical(spsn: list) -> np.ndarray:
    """produces the confusion matrix of a clinical modality, i.e. a modality
    that can not detect microscopic metastases

    Args:
        spsn (list): list with specificity and sensitivity of modality

    Returns:
        np.ndarray: confusion matrix of modality
    """
    try:
        check_spsn(spsn)
    except ValueError:
        raise
    sp, sn = spsn
    confusion_matrix = np.array([[sp, 1. - sp],
                                 [sp, 1. - sp],
                                 [1. - sn, sn]])
    return confusion_matrix


def pathological(spsn: list) -> np.ndarray:
    """produces the confusion matrix of a pathological modality, i.e. a modality
    that can detect microscopic metastases

    Args:
        spsn (list): list with specificity and sensitivity of modality

    Returns:
        np.ndarray: confusion matrix of modality
    """
    try:
        check_spsn(spsn)
    except ValueError:
        raise
    sp, sn = spsn
    confusion_matrix = np.array([[sp, 1. - sp],
                                 [1. - sn, sn],
                                 [1. - sn, sn]])
    return confusion_matrix
