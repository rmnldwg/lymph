"""
Methods & classes to manage matrices of the :py:class:`~lymph.models.Unilateral` class.
"""
# pylint: disable=too-few-public-methods
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from cachetools import LRUCache

from lymph import models
from lymph.helper import (
    AbstractLookupDict,
    arg0_cache,
    get_state_idx_matrix,
    row_wise_kron,
    tile_and_repeat,
)


def generate_transition(instance: models.Unilateral) -> np.ndarray:
    """Compute the transition matrix of the lymph model."""
    lnls_list = list(instance.graph.lnls.values())
    num_lnls = len(lnls_list)
    num_states = 3 if instance.graph.is_trinary else 2
    transition_matrix = np.ones(shape=(num_states**num_lnls, num_states**num_lnls))

    for i, lnl in enumerate(lnls_list):
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
                parent_node_i = lnls_list.index(edge.parent)
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


cached_generate_transition = arg0_cache(maxsize=128, cache_class=LRUCache)(generate_transition)
"""Cached version of :py:func:`generate_transition`.

This expects the first argument to be a hashable object that is used instrad of the
``instance`` argument of :py:func:`generate_transition`. It is intended to be used with
the :py:meth:`~lymph.graph.Representation.parameter_hash` method of the graph.
"""


def generate_observation(instance: models.Unilateral) -> np.ndarray:
    """Generate the observation matrix of the lymph model."""
    num_lnls = len(instance.graph.lnls)
    base = 2 if instance.graph.is_binary else 3
    shape = (base ** num_lnls, 1)
    observation_matrix = np.ones(shape=shape)

    for modality in instance.modalities.values():
        mod_obs_matrix = np.ones(shape=(1,1))
        for _ in instance.graph.lnls:
            mod_obs_matrix = np.kron(mod_obs_matrix, modality.confusion_matrix)

        observation_matrix = row_wise_kron(observation_matrix, mod_obs_matrix)

    return observation_matrix


cached_generate_observation = arg0_cache(maxsize=128, cache_class=LRUCache)(generate_observation)
"""Cached version of :py:func:`generate_observation`.

This expects the first argument to be a hashable object that is used instrad of the
``instance`` argument of :py:func:`generate_observation`. It is intended to be used
with the hash of all confusion matrices of the model's modalities, which is returned
by the method :py:meth:`~lymph.modalities.ModalitiesUserDict.confusion_matrices_hash`.
"""


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

    Examples:
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


def generate_data_encoding(model: models.Unilateral, t_stage: str) -> np.ndarray:
    """Generate the data matrix for a specific T-stage from patient data.

    The :py:attr:`~lymph.models.Unilateral.patient_data` needs to contain the column
    ``"_model"``, which is constructed when loading the data into the model. From this,
    a data matrix is constructed for the given ``t_stage``.

    The returned matrix has the shape :math:`2^{N \\cdot \\mathcal{O}} \\times M`,
    where :math:`N` is the number of lymph node levels, :math:`\\mathcal{O}` is the
    number of diagnostic modalities and :math:`M` is the number of patients with the
    given ``t_stage``.
    """
    if not model.patient_data["_model", "#", "t_stage"].isin([t_stage]).any():
        raise ValueError(f"No patients with T-stage {t_stage} in patient data.")

    has_t_stage = model.patient_data["_model", "#", "t_stage"] == t_stage
    patients_with_t_stage = model.patient_data[has_t_stage]

    result = np.ones(
        shape=(model.observation_matrix().shape[1], len(patients_with_t_stage)),
        dtype=bool,
    )

    for i, (_, patient_row) in enumerate(patients_with_t_stage["_model"].iterrows()):
        patient_encoding = np.ones(shape=1, dtype=bool)
        for modality_name in model.modalities.keys():
            if modality_name not in patient_row:
                continue
            diagnose_encoding = compute_encoding(
                lnls=model.graph.lnls.keys(),
                pattern=patient_row[modality_name],
                base=2,   # observations are always binary!
            )
            patient_encoding = np.kron(patient_encoding, diagnose_encoding)

        result[:,i] = patient_encoding

    return result


class DataEncodingUserDict(AbstractLookupDict):
    """``UserDict`` that dynamically generates the data matrices for each T-stage.

    The data matrix is a binary encoding of all complete diagnoses that are plausible
    for an actual patient with possibly incomplete information. So, every columns may
    contain multiple ones, each for a complete diagnosis that could have led to the
    observed data.

    When multiplying a distribution over possible complete observations with the data
    matrix, the result is essentially an array of likelihoods for each patient.

    See Also:
        :py:attr:`~lymph.models.Unilateral.data_matrices`
    """
    model: models.Unilateral

    def __setitem__(self, __key, __value) -> None:
        warnings.warn("Setting the data matrices is not supported.")

    def __missing__(self, t_stage: str):
        """Create the data matrix for a specific T-stage if necessary."""
        try:
            self.data[t_stage] = generate_data_encoding(self.model, t_stage)
        except ValueError as val_err:
            raise KeyError(f"No data matrix for T-stage {t_stage}.") from val_err

        return self[t_stage]


def generate_diagnose(model: models.Unilateral, t_stage: str) -> np.ndarray:
    """Generate the diagnose matrix for a specific T-stage.

    The diagnose matrix is the product of the observation matrix and the data matrix
    for the given ``t_stage``.
    """
    return model.observation_matrix() @ model.data_matrices[t_stage]


cached_generate_diagnose = arg0_cache(maxsize=128, cache_class=LRUCache)(generate_diagnose)
"""Cached version of :py:func:`generate_diagnose`.

The decorated function expects an additional first argument that should be unique for
the combination of modalities and patient data. It is intended to be used with the
joint hash of the modalities
(:py:meth:`~lymph.modalities.ModalitiesUserDict.confusion_matrices_hash`) and the
patient data hash that is always precomputed when a new dataset is loaded into the
model (:py:meth:`~lymph.models.Unilateral.patient_data_hash`).
"""


class DiagnoseUserDict(AbstractLookupDict):
    """``UserDict`` that dynamically generates the diagnose matrices for each T-stage.

    A diagnose matrix for a T-stage is simply the product of the observation matrix and
    the data matrix for that T-stage. Precomputing and caching this matrix is useful,
    because it does not depend on the model parameters and can be reused until either
    the diagnostic modalities (meaning the observation matrix needs to be updated) or
    the patient data (meaning the data matrix needs to be updated) change.

    See Also:
        :py:attr:`~lymph.models.Unilateral.diagnose_matrices`
    """
    model: models.Unilateral

    def __setitem__(self, __key, __value) -> None:
        warnings.warn("Setting the diagnose matrices is not supported.")

    def __getitem__(self, key: Any) -> Any:
        modalities_hash = self.model.modalities.confusion_matrices_hash()
        patient_data_hash = self.model.patient_data_hash
        joint_hash = hash((modalities_hash, patient_data_hash, key))
        return cached_generate_diagnose(joint_hash, self.model, key)

    def __missing__(self, t_stage: str):
        """Create the diagnose matrix for a specific T-stage if necessary."""
        return self[t_stage]
