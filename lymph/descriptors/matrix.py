"""Descriptors to manage the matrices of the :py:class:`~models.Unilateral` class."""
# pylint: disable=too-few-public-methods
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from lymph import models
from lymph.descriptors import AbstractDictDescriptor, AbstractLookupDict
from lymph.helper import get_state_idx_matrix, row_wise_kron, tile_and_repeat


class AbstractMatrixDescriptor:
    """Abstract descriptor class to lazily and dynamically compute a matrix attribute.

    Many matrices that the :py:class:`~models.Unilateral` class uses are expensive
    to compute. This descriptor class allows to lazily compute the matrix dynamically
    when it is needed for the first time after a change, e.g. in the parameters, has
    occurred.
    """
    def __set_name__(self, owner, name: str):
        self.private_name = '_' + name


    def __get__(self, instance: models.Unilateral, _cls) -> np.ndarray:
        """Return the matrix of the lymph model."""
        if not hasattr(instance, self.private_name):
            matrix = self.generate(instance)
            setattr(instance, self.private_name, matrix)

        return getattr(instance, self.private_name)


    def __delete__(self, instance: models.Unilateral):
        """Delete the matrix of the lymph model."""
        if hasattr(instance, self.private_name):
            delattr(instance, self.private_name)


    @staticmethod
    def generate(instance: models.Unilateral) -> np.ndarray:
        """Compute the matrix of the lymph model."""
        raise NotImplementedError


class Transition(AbstractMatrixDescriptor):
    """Descriptor class to access the transition matrix of a lymph model.

    The transition matrix describes the probability of transitioning from one possible
    hidden state to another. In any hidden Markov model, the transition matrix is
    the crucial component that describes the dynamics of the model. It is square with
    the number of rows and columns corresponding to the number of hidden states and its
    rows sum to one.

    When first trying to access this descriptor, it will compute the transition
    matrix of the model. This is done by iterating over all edges and creating
    a matrix with rows and columns corresponding to the nodes of the model.

    When any of the parameters that influence the transition matrix are changed, the
    transition matrix should be deleted. This will trigger a re-computation of the
    transition matrix when it is needed next.

    See Also:
        :py:class:`~lymph.descriptors.params.GetterSetterAccess`
            This descriptor manages the lookup for all parameter getter and setter
            methods that are stored in a small
            :py:class:`~lymph.descriptors.params.Param` object. This is also where the
            transition matrix is deleted when a parameter is set.
    """
    @staticmethod
    def generate(instance: models.Unilateral):
        """Compute the transition matrix of the lymph model."""
        num_lnls = len(instance.graph._lnls)
        num_states = 3 if instance.graph.is_trinary else 2
        transition_matrix = np.ones(shape=(num_states**num_lnls, num_states**num_lnls))

        for i, lnl in enumerate(instance.graph._lnls):
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
                    parent_node_i = instance.graph._lnls.index(edge.parent)
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


class Observation(AbstractMatrixDescriptor):
    """Descriptor class to access the observation matrix of a lymph model.

    The observation matrix describes the probability of observing a certain diagnosis
    given the state of the model. A diagnosis refers in this case to a complete set
    of observations for all lymph node levels for every diagnostic modality. The
    observation matrix is rectangular with the number of rows corresponding to the
    number of possible hidden states and the number of columns corresponding to the
    number of possible complete diagnoses. The rows of the observation matrix sum to
    one.

    Multiplying the distribution over hidden states with the observation matrix
    results in the distribution over diagnoses. This is the key step in the computation
    of the likelihood of a diagnosis given the lymph model.

    When first trying to access this descriptor, it will compute the observation
    matrix of the model from the diagnostic modalities provided.

    See Also:
        :py:class:`lymph.descriptors.modalities`
            This module manages the lookup for all diagnostic modalities and their
            confusion matrices from which the observation matrix is computed.
    """
    @staticmethod
    def generate(instance: models.Unilateral) -> np.ndarray:
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


def compute_encoding(
    lnls: list[str],
    pattern: pd.Series | dict[str, bool],
) -> np.ndarray:
    """Compute the binary encoding of a particular ``pattern`` of involvement.

    Here, a ``pattern`` must be indexable with the LNL names in ``lnls`` and return
    either ``True``, if the respective LNL is involved, or ``False`` if it is healthy.
    If it contains anything that ``pd.isna()`` considers ``True``, the respective
    LNL is considered to be unknown.
    """
    num_lnls = len(lnls)
    encoding = np.ones(shape=2 ** num_lnls, dtype=bool)

    for j, lnl in enumerate(lnls):
        if lnl not in pattern or pd.isna(pattern[lnl]):
            continue
        encoding = np.logical_and(
            encoding,
            tile_and_repeat(
                mat=np.array([not pattern[lnl], pattern[lnl]]),
                tile=(1, 2 ** j),
                repeat=(1, 2 ** (num_lnls - j - 1)),
            )[0],
        )
    return encoding


def generate_data_matrix(model: models.Unilateral, t_stage: str) -> np.ndarray:
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
        shape=(model.observation_matrix.shape[1], len(patients_with_t_stage)),
        dtype=bool,
    )

    for i, (_, patient_row) in enumerate(patients_with_t_stage["_model"].iterrows()):
        patient_encoding = np.ones(shape=1, dtype=bool)
        for modality_name in model.modalities.keys():
            if modality_name not in patient_row:
                continue
            diagnose_encoding = compute_encoding(
                lnls=[lnl.name for lnl in model.graph._lnls],
                pattern=patient_row[modality_name],
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
    def __setitem__(self, __key, __value) -> None:
        warnings.warn("Setting the data matrices is not supported.")

    def __missing__(self, t_stage: str):
        """Create the data matrix for a specific T-stage if necessary."""
        try:
            self.data[t_stage] = generate_data_matrix(self.model, t_stage)
        except ValueError as val_err:
            raise KeyError(f"No data matrix for T-stage {t_stage}.") from val_err

        return self[t_stage]


class DataEncodings(AbstractDictDescriptor):
    """Allows accessing the data matrix of every T-category separately."""
    def _get_callback(self, instance: models.Unilateral):
        """Add ``instance`` as ``model`` attribute to the descriptor."""
        data_encoding_dict = DataEncodingUserDict(model=instance)
        setattr(instance, self.private_name, data_encoding_dict)

    def __set__(self, instance, value):
        raise AttributeError("Cannot set data matrix lookup dict.")


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
    def __setitem__(self, __key, __value) -> None:
        warnings.warn("Setting the diagnose matrices is not supported.")

    def __missing__(self, t_stage: str) -> np.ndarray:
        """If the matrix for a ``t_stage`` is missing, try to generate it lazily."""
        self.data[t_stage] = (
            self.model.observation_matrix @ self.model.data_matrices[t_stage]
        )
        return self[t_stage]


class Diagnoses(AbstractDictDescriptor):
    """Descriptor that instantiates and manages the :py:class:`~DiagnoseUserDict`."""
    def _get_callback(self, instance: models.Unilateral):
        """Add ``instance`` as ``model`` attribute to the descriptor."""
        diagnose_dict = DiagnoseUserDict(model=instance)
        setattr(instance, self.private_name, diagnose_dict)

    def __set__(self, instance, value):
        raise AttributeError("Cannot set diagnose matrices lookup dict.")