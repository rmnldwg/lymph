"""
Methods & classes to manage matrices of the :py:class:`~lymph.models.Unilateral` class.
"""
# pylint: disable=too-few-public-methods
from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd

from lymph import graph
from lymph.utils import get_state_idx_matrix, row_wise_kron, tile_and_repeat
from lymph.modalities import Modality


@lru_cache(maxsize=128)
def generate_transition(
    lnls: Iterable[graph.LymphNodeLevel],
    num_states: int,
) -> np.ndarray:
    """Compute the transition matrix of the lymph model."""
    lnls = list(lnls)   # necessary for `index()` call
    num_lnls = len(lnls)
    transition_matrix = np.ones(shape=(num_states**num_lnls, num_states**num_lnls))

    for i, lnl in enumerate(lnls):
        current_state_idx = get_state_idx_matrix(
            lnl_idx=i,
            num_lnls=num_lnls,
            num_states=num_states,
        )
        new_state_idx = current_state_idx.T

        # This needs to be initialized with a one where no transition happens
        # and a zero where a transition happens. This is because of how differently
        # the transition matrix entries are computed for no spread vs. spread.
        lnl_transition_matrix = new_state_idx == current_state_idx

        for edge in lnl.inc:
            if edge.is_tumor_spread:
                edge_transition_grid = edge.transition_tensor[
                    0, current_state_idx, new_state_idx
                ]
            else:
                parent_node_i = lnls.index(edge.parent)
                parent_state_idx = get_state_idx_matrix(
                    lnl_idx=parent_node_i,
                    num_lnls=num_lnls,
                    num_states=num_states,
                )
                edge_transition_grid = edge.transition_tensor[
                    parent_state_idx, current_state_idx, new_state_idx
                ]

            lnl_transition_matrix = np.where(
                # For transitions, we need to compute the probability that none of
                # the incoming edges of an LNL spread. This is done by multiplying
                # the probabilities of all edges not spreading and taking the
                # complement: 1 - (1 - p_1) * (1 - p_2) * ...
                # If an LNL remains in its current state, the probability is
                # simply the product of all incoming edges not spreading.
                new_state_idx == current_state_idx + 1,
                1 - (1 - lnl_transition_matrix) * (1 - edge_transition_grid),
                lnl_transition_matrix * edge_transition_grid,
            )

        transition_matrix *= lnl_transition_matrix

    return transition_matrix


@lru_cache(maxsize=128)
def generate_observation(
    modalities: Iterable[Modality],
    num_lnls: int,
    base: int = 2,
) -> np.ndarray:
    """Generate the observation matrix of the lymph model."""
    shape = (base ** num_lnls, 1)
    observation_matrix = np.ones(shape=shape)

    for modality in modalities:
        mod_obs_matrix = np.ones(shape=(1,1))
        for _ in range(num_lnls):
            mod_obs_matrix = np.kron(mod_obs_matrix, modality.confusion_matrix)

        observation_matrix = row_wise_kron(observation_matrix, mod_obs_matrix)

    return observation_matrix


def compute_encoding(
    lnls: list[str],
    pattern: pd.Series | dict[str, bool | int | str],
    base: int = 2,
) -> np.ndarray:
    """Compute the encoding of a particular ``pattern`` of involvement.

    A ``pattern`` holds information about the involvement of each LNL and the function
    transforms this into a binary encoding which is ``True`` for all possible complete
    states/diagnoses that are compatible with the given ``pattern``.

    In the binary case (``base=2``), the value behind ``pattern[lnl]`` can be one of
    the following things:
    - ``False``: The LNL is healthy.
    - ``"healthy"``: The LNL is healthy.
    - ``True``: The LNL is involved.
    - ``"involved"``: The LNL is involved.
    - ``pd.isna(pattern[lnl]) == True``: The involvement of the LNL is unknown.

    In the trinary case (``base=3``), the value behind ``pattern[lnl]`` can be one of
    these things:
    - ``False``: The LNL is healthy.
    - ``"healthy"``: The LNL is healthy.
    - ``True``: The LNL is involved (micro- or macroscopic).
    - ``"involved"``: The LNL is involved (micro- or macroscopic).
    - ``"micro"``: The LNL is involved microscopically only.
    - ``"macro"``: The LNL is involved macroscopically only.
    - ``"notmacro"``: The LNL is healthy or involved microscopically.

    Missing values are treated as unknown involvement.

    >>> compute_encoding(["II", "III"], {"II": True, "III": False})
    array([False, False,  True, False])
    >>> compute_encoding(["II", "III"], {"II": "involved"})
    array([False, False,  True,  True])
    >>> compute_encoding(
    ...     lnls=["II", "III"],
    ...     pattern={"II": True, "III": False},
    ...     base=3,
    ... )
    array([False, False, False,  True, False, False,  True, False, False])
    >>> compute_encoding(
    ...     lnls=["II", "III"],
    ...     pattern={"II": "micro", "III": "notmacro"},
    ...     base=3,
    ... )
    array([False, False, False,  True,  True, False, False, False, False])
    """
    num_lnls = len(lnls)
    encoding = np.ones(shape=base ** num_lnls, dtype=bool)

    if base == 2:
        element_map = {
            "healthy": np.array([True, False]),
            False: np.array([True, False]),
            "involved": np.array([False, True]),
            True: np.array([False, True]),
        }
    elif base == 3:
        element_map = {
            "healthy": np.array([True, False, False]),
            False: np.array([True, False, False]),
            "involved": np.array([False, True, True]),
            True: np.array([False, True, True]),
            "micro": np.array([False, True, False]),
            "macro": np.array([False, False, True]),
            "notmacro": np.array([True, True, False]),
        }
    else:
        raise ValueError(f"Invalid base {base}.")

    for j, lnl in enumerate(lnls):
        if lnl not in pattern or pd.isna(pattern[lnl]):
            continue

        try:
            element = element_map[pattern[lnl]]
        except KeyError as key_err:
            raise ValueError(
                f"Invalid pattern for LNL {lnl}: {pattern[lnl]}",
            ) from key_err

        encoding = np.logical_and(
            encoding,
            tile_and_repeat(
                mat=element,
                tile=(1, base ** j),
                repeat=(1, base ** (num_lnls - j - 1)),
            )[0],
        )
    return encoding


def generate_data_encoding(
    patient_data: pd.DataFrame,
    modalities: dict[str, Modality],
    lnls: list[str],
) -> np.ndarray:
    """Generate the data matrix for a specific T-stage from patient data.

    The :py:attr:`.models.Unilateral.patient_data` needs to contain the column
    ``"_model"``, which is constructed when loading the data into the model. From this,
    a data matrix is constructed for all present diagnostic modalities.

    The returned matrix has the shape :math:`2^{N \\cdot \\mathcal{O}} \\times M`,
    where :math:`N` is the number of lymph node levels, :math:`\\mathcal{O}` is the
    number of diagnostic modalities and :math:`M` is the number of patients.
    """
    result = np.ones(
        shape=(2 ** (len(lnls) * len(modalities)), len(patient_data)),
        dtype=bool,
    )

    for i, (_, patient_row) in enumerate(patient_data["_model"].iterrows()):
        patient_encoding = np.ones(shape=1, dtype=bool)
        for modality_name in modalities.keys():
            if modality_name not in patient_row:
                warnings.warn(f"Modality {modality_name} not in data. Skipping.")
                continue
            diagnose_encoding = compute_encoding(
                lnls=lnls,
                pattern=patient_row[modality_name],
                base=2,   # observations are always binary!
            )
            patient_encoding = np.kron(patient_encoding, diagnose_encoding)

        result[:,i] = patient_encoding

    return result.T


@lru_cache
def evolve_midext(max_time: int, midext_prob: int) -> np.ndarray:
    """Compute the evolution over the state of a tumor's midline extension."""
    midext_states = np.zeros(shape=(max_time + 1, 2), dtype=float)
    midext_states[0,0] = 1.

    midext_transition_matrix = np.array([
        [1 - midext_prob, midext_prob],
        [0.             , 1.         ],
    ])

    # compute midext prob for all time steps
    for i in range(len(midext_states) - 1):
        midext_states[i+1,:] = midext_states[i,:] @ midext_transition_matrix

    return midext_states


def fast_trace(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Compute the trace of a product of two matrices (``left`` and ``right``).

    This is based on the observation that the trace of a product of two matrices is
    equal to the sum of the element-wise products of the two matrices. See
    `Wikipedia <https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Properties>`_ and
    `StackOverflow <https://stackoverflow.com/a/18854776>`_ for more information.
    """
    return np.sum(left.T * right, axis=0)
