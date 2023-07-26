"""Descriptor to manage the transition from state to state in the HMM context."""
from __future__ import annotations

import numpy as np

from lymph import models
from lymph.helper import get_state_idx_matrix


class AbstractMatrixDescriptor:
    """Abstract descriptor class to access a matrix of a lymph model.

    The matrix describes the probability of transitioning from one state to another
    or the probability of observing a certain diagnosis given the state of the model.

    When first trying to access this descriptor, it will compute the matrix of the
    model.
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


    def generate(self, instance: models.Unilateral) -> np.ndarray:
        """Compute the matrix of the lymph model."""
        raise NotImplementedError


class Transition(AbstractMatrixDescriptor):
    """Descriptor class to access the transition matrix of a lymph model.

    The transition matrix describes the probability of transitioning from one state
    to another.

    When first trying to access this descriptor, it will compute the transition
    matrix of the model. This is done by iterating over all edges and creating
    a matrix with rows and columns corresponding to the nodes of the model.
    """
    @staticmethod
    def generate(instance: models.Unilateral):
        """Compute the transition matrix of the lymph model."""
        num_lnls = len(instance.lnls)
        num_states = 3 if instance.is_trinary else 2
        transition_matrix = np.ones(shape=(num_states**num_lnls, num_states**num_lnls))

        for i, lnl in enumerate(instance.lnls):
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
                    parent_node_i = instance.lnls.index(edge.parent)
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
    given the state of the model.

    When first trying to access this descriptor, it will compute the observation
    matrix of the model from the diagnostic modalities provided.
    """
    @staticmethod
    def generate(instance: models.Unilateral) -> np.ndarray:
        """Generate the observation matrix of the lymph model."""
        num_lnls = len(instance.lnls)
        num_states = len(instance.allowed_states)
        observation_matrix = np.ones(shape=(len(instance.state_list), 2**num_lnls))
        for i, confusion_matrix in enumerate(instance.modalities.values()):
            tiled = np.tile(confusion_matrix, (num_states**i, 2**i))
            repeat_along_0 = np.repeat(
                tiled,
                num_states**(len(instance.modalities) - i - 1),
                axis=0,
            )
            modality_obs_mat = np.repeat(
                repeat_along_0,
                2**(len(instance.modalities) - i - 1),
                axis=1,
            )
            observation_matrix *= modality_obs_mat

        return observation_matrix
