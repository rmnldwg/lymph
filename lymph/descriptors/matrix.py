"""Descriptor to manage the transition from state to state in the HMM context."""
from __future__ import annotations

import numpy as np

from lymph import models
from lymph.helper import get_state_idx_matrix


class Transition:
    """Descriptor class to access the transition matrix of a lymph model.

    When first trying to access this descriptor, it will compute the transition
    matrix of the model. This is done by iterating over all edges and creating
    a matrix with rows and columns corresponding to the nodes of the model.
    """
    def __set_name__(self, owner, name: str):
        self.private_name = '_' + name


    def __get__(self, instance: models.Unilateral, _cls) -> np.ndarray:
        """Return the transition matrix of the lymph model."""
        if not hasattr(instance, self.private_name):
            self.generate_system_transition_matrix(instance)

        return getattr(instance, self.private_name)


    def __delete__(self, instance: models.Unilateral):
        """Delete the transition matrix of the lymph model."""
        if hasattr(instance, self.private_name):
            delattr(instance, self.private_name)


    def generate_system_transition_matrix(self, instance: models.Unilateral):
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
                    parent_node_i = instance.lnls.index(edge.start)
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

        setattr(instance, self.private_name, transition_matrix)
