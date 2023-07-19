"""Descriptor to manage the transition from state to state in the HMM context."""
import numpy as np


class Matrix:
    """Descriptor class to access the transition matrix of a lymph model.

    When first trying to access this descriptor, it will compute the transition
    matrix of the model. This is done by iterating over all edges and creating
    a matrix with rows and columns corresponding to the nodes of the model.
    """
    def __set_name__(self, owner, name: str):
        self.private_name = '_' + name


    def __get__(self, instance, _cls) -> np.ndarray:
        """Return the transition matrix of the lymph model."""
        if not hasattr(instance, self.private_name):
            self.generate_transition_matrix(instance)

        return getattr(instance, self.private_name)


    def __delete__(self, instance):
        """Delete the transition matrix of the lymph model."""
        if hasattr(instance, self.private_name):
            delattr(instance, self.private_name)


    def generate_transition_matrix(self, instance):
        """Compute the transition matrix of the lymph model."""
        num = len(instance.state_list)
        transition_matrix = np.zeros(shape=(num, num))

        for i, state in enumerate(instance.state_list):
            instance.set_state(state)
            for j in instance.allowed_transitions[i]:
                new_state = instance.state_list[j]
                transition_prob = instance.comp_transition_prob(new_state)
                transition_matrix[i, j] = transition_prob

        setattr(instance, self.private_name, transition_matrix)


class Mask:
    """Descriptor class to manage the mask of allowed state transitions.

    Because we prohibit self-healing, a large portion of the transition matrix contains
    zeros. This descriptor creates a mask that is used when generating the transition
    matrix, so that only allowed transitions are computed.
    """
    def __set_name__(self, owner, name: str):
        self.private_name = '_' + name


    def __get__(self, instance, _cls) -> np.ndarray:
        """Return the mask of allowed state transitions."""
        if not hasattr(instance, self.private_name):
            self.generate_mask(instance)

        return getattr(instance, self.private_name)


    def generate_mask(self, instance):
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
