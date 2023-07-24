"""Module containing supporting classes and functions."""

import doctest
from functools import lru_cache
import warnings
from typing import Optional, List

import numpy as np


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
        raise TypeError("Modality names must be strings.")

    has_len_2 = len(spsn) == 2
    is_above_lb = np.all(np.greater_equal(spsn, 0.5))
    is_below_ub = np.all(np.less_equal(spsn, 1.))

    if not has_len_2 or not is_above_lb or not is_below_ub:
        raise ValueError(
            "For each modality provide a list of two decimals between 0.5 and 1.0 "
            "as specificity & sensitivity respectively."
        )


def clinical(spsn: list) -> np.ndarray:
    """produces the confusion matrix of a clinical modality, i.e. a modality
    that can not detect microscopic metastases

    Args:
        spsn (list): list with specificity and sensitivity of modality

    Returns:
        np.ndarray: confusion matrix of modality
    """
    check_spsn(spsn)
    sp, sn = spsn
    confusion_matrix = np.array([
        [sp     , 1. - sp],
        [sp     , 1. - sp],
        [1. - sn, sn     ],
    ])
    return confusion_matrix


def pathological(spsn: list) -> np.ndarray:
    """produces the confusion matrix of a pathological modality, i.e. a modality
    that can detect microscopic metastases

    Args:
        spsn (list): list with specificity and sensitivity of modality

    Returns:
        np.ndarray: confusion matrix of modality
    """
    check_spsn(spsn)
    sp, sn = spsn
    confusion_matrix = np.array([
        [sp     , 1. - sp],
        [1. - sn, sn     ],
        [1. - sn, sn     ],
    ])
    return confusion_matrix


def tile_and_repeat(mat: np.ndarray, i: int, num: int) -> np.ndarray:
    """Tile and repeat a matrix.

    The matrix `mat` is first tiled 2**i times, then repeated 2**(num-i) times along
    both axes.

    Example:
    >>> mat = np.array([[1, 2], [3, 4]])
    >>> tile_and_repeat(mat, 1, 2)
    array([[1, 1, 2, 2, 1, 1, 2, 2],
           [1, 1, 2, 2, 1, 1, 2, 2],
           [3, 3, 4, 4, 3, 3, 4, 4],
           [3, 3, 4, 4, 3, 3, 4, 4],
           [1, 1, 2, 2, 1, 1, 2, 2],
           [1, 1, 2, 2, 1, 1, 2, 2],
           [3, 3, 4, 4, 3, 3, 4, 4],
           [3, 3, 4, 4, 3, 3, 4, 4]])
    """
    tiled = np.tile(mat, (2**i, 2**i))
    repeat_along_0 = np.repeat(tiled, 2**(num-i), axis=0)
    return np.repeat(repeat_along_0, 2**(num-i), axis=1)


@lru_cache
def get_state_idx_matrix(lnl_idx: int, num_lnls: int, num_states: int) -> np.ndarray:
    """Return the indices for the transition tensor correpsonding to `lnl_idx`.

    Example:
    >>> get_state_idx_matrix(1, 3, 2)
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]])
    >>> get_state_idx_matrix(1, 2, 3)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 2, 2, 2]])
    """
    indices = np.arange(num_states).reshape(num_states, -1)
    row = np.tile(indices, (num_states ** lnl_idx, num_states ** num_lnls))
    return np.repeat(row, num_states ** (num_lnls - lnl_idx - 1), axis=0)


if __name__ == "__main__":
    doctest.testmod()
