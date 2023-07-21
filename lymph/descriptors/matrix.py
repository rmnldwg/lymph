"""Descriptor to manage the transition from state to state in the HMM context."""
from __future__ import annotations

import numpy as np

from lymph import models
from lymph.helper import get_state_idx_matrix, init_recursively_upper_tri


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
        transition_matrix = init_recursively_upper_tri(num_lnls)

        for i, lnl in enumerate(instance.lnls):
            lnl_transition_matrix = init_recursively_upper_tri(num_lnls)

            current_state_idx = get_state_idx_matrix(i, num_lnls)
            new_state_idx = current_state_idx.T

            for j, edge in enumerate(lnl.inc):
                if edge.is_tumor_spread:
                    parent_state_idx = np.ones(shape=(2**num_lnls, 2**num_lnls))
                    edge_transition_grid = edge.transition_tensor[
                        0, current_state_idx, new_state_idx
                    ]
                else:
                    parent_node_i = instance.lnls.index(edge.start)
                    parent_state_idx = get_state_idx_matrix(parent_node_i, num_lnls)
                    edge_transition_grid = edge.transition_tensor[
                        parent_state_idx, current_state_idx, new_state_idx
                    ]

                if j == 0:
                    lnl_transition_matrix *= edge_transition_grid
                else:
                    lnl_transition_matrix = np.where(
                        (
                            # if this is true, then we need to compute the probability of
                            # spread based on the original transition matrix's value t OR
                            # because of what the edge adds to this (e). So, we compute
                            # 1 - (1 - t) * (1 - e).
                            # If it is false, it means we compute the probability of NO spread
                            # and we can just multiply the two values: t * e (they are not the
                            # same).
                            (parent_state_idx == 1)
                            & (current_state_idx == 0)
                            & (new_state_idx == 1)
                        ),
                        1 - (1 - lnl_transition_matrix) * (1 - edge_transition_grid),
                        lnl_transition_matrix * edge_transition_grid,
                    )

            transition_matrix *= lnl_transition_matrix

        setattr(instance, self.private_name, transition_matrix)


class TransitionMask:
    """Descriptor class to manage the mask of allowed state transitions.

    Because we prohibit self-healing, a large portion of the transition matrix contains
    zeros. This descriptor creates a mask that is used when generating the transition
    matrix, so that only allowed transitions are computed.
    """
    def __set_name__(self, owner, name: str):
        self.private_name = '_' + name


    def __get__(self, instance: models.Unilateral, _cls) -> np.ndarray:
        """Return the mask of allowed state transitions."""
        if not hasattr(instance, self.private_name):
            self.generate_mask(instance)

        return getattr(instance, self.private_name)


    def generate_mask(self, instance: models.Unilateral):
        """Compute the mask of allowed state transitions."""
        mask = {}

        for i in range(len(instance.state_list)):
            mask[i] = []
            for j in range(len(instance.state_list)):
                is_self_healing = np.any(np.greater(
                    instance.state_list[i,:],
                    instance.state_list[j,:],
                ))
                if not is_self_healing:
                    mask[i].append(j)

        setattr(instance, self.private_name, mask)
