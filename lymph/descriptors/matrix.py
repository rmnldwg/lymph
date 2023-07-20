"""Descriptor to manage the transition from state to state in the HMM context."""
from __future__ import annotations

import numpy as np

from lymph import models
from lymph.helper import get_transition_tensor_idx


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
        num_states = len(instance.state_list)
        num_lnls = len(instance.lnls)
        transition_matrix = np.ones(shape=(num_states, num_states))

        for end_node_i, lnl in enumerate(instance.lnls):
            end_node_idx = get_transition_tensor_idx(end_node_i, num_lnls)
            new_state_idx = end_node_idx.T
            for edge in lnl.inc:
                if edge.is_tumor_spread:
                    start_node_idx = 0
                else:
                    start_node_i = instance.nodes.index(edge.start)
                    start_node_idx = get_transition_tensor_idx(start_node_i, num_lnls)
                transition_matrix *= edge.transition_tensor[
                    start_node_idx, end_node_idx, new_state_idx
                ]

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
