"""
Module defining the nodes of the graph representing the lymphatic system.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class AbstractNode:
    """
    Abstract base class for nodes in the graph reprsenting the lymphatic system.
    """
    def __init__(
        self,
        name: str,
        state: int,
        allowed_states: Optional[List[int]] = None,
    ) -> None:
        """
        Make a new node.

        Upon initialization, the `name` and `state` of the node must be provided. The
        `state` must be one of the `allowed_states`. The constructor makes sure that
        the `allowed_states` are a list of ints, even when, e.g., a tuple of floats
        is provided.
        """
        self.name = name

        if allowed_states is None:
            allowed_states = [0, 1]

        _allowed_states = []
        for s in allowed_states:
            try:
                _allowed_states.append(int(s))
            except ValueError as val_err:
                raise ValueError("Allowed states must be castable to int") from val_err

        self.allowed_states = _allowed_states
        self.state = state

        # nodes can have outgoing edge connections
        self.out = []


    def __str__(self) -> str:
        """Return a string representation of the node."""
        return self.name


    @property
    def name(self) -> str:
        """Return the name of the node."""
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """Set the name of the node."""
        try:
            new_name = str(new_name)
        except ValueError as val_err:
            raise ValueError("Name of node must be castable to string") from val_err

        self._name = new_name


    @property
    def state(self) -> int:
        """Return the state of the node."""
        return self._state

    @state.setter
    def state(self, new_state: int) -> None:
        """Set the state of the node."""
        try:
            new_state = int(new_state)
        except ValueError as val_err:
            raise ValueError("State of node must be castable to int") from val_err

        if new_state not in self.allowed_states:
            raise ValueError("State of node must be one of the allowed states")

        self._state = new_state


    def comp_bayes_net_prob(self, log: bool = False) -> float:
        """Compute the Bayesian network's probability for the current state."""
        return 0. if log else 1.


    def comp_trans_prob(self, new_state: int, log: bool = False) -> float:
        """Compute the hidden Markov model's transition probability to a new state."""
        if new_state not in self.allowed_states:
            raise ValueError("New state must be one of the allowed states")

        if new_state < self.state:
            return -np.inf if log else 0.

        return 0. if log else 1.


    def comp_obs_prob(
        self,
        obs: int,
        obs_table: np.ndarray,
        log: bool = False,
    ) -> float:
        """Compute the probability of the diagnosis `obs`, given the current state.

        The `obs_table` is a 2D array with the rows corresponding to the states and
        the columns corresponding to the observations. It encodes for each state and
        diagnosis the corresponding probability.
        """
        if obs is None or np.isnan(obs):
            return 0 if log else 1.
        obs_prob = obs_table[self.state, int(obs)]
        return np.log(obs_prob) if log else obs_prob


class Tumor(AbstractNode):
    """A tumor in the graph representation of the lymphatic system."""
    def __init__(self, name: str, state: int) -> None:
        """Create a new tumor.

        A tumor can only ever be in one state, and it cannot change its state.
        """
        allowed_states = [state]
        super().__init__(name, state, allowed_states)


    def __str__(self):
        """Print basic info"""
        return f"Tumor {super().__str__()}"


class LymphNodeLevel(AbstractNode):
    """A lymph node level (LNL) in the graph representation of the lymphatic system."""
    def __init__(
        self,
        name: str,
        state: int = 0,
        allowed_states: Optional[List[int]] = None,
    ) -> None:
        """Create a new lymph node level."""

        super().__init__(name, state, allowed_states)

        # LNLs can also have incoming edge connections
        self.inc = []


    def __str__(self):
        """Print basic info"""
        return f"LNL {super().__str__()}"


    @property
    def is_binary(self) -> bool:
        """Return whether the node is binary."""
        return len(self.allowed_states) == 2


    @property
    def is_trinary(self) -> bool:
        """Return whether the node is trinary."""
        return len(self.allowed_states) == 3


    def comp_bayes_net_prob(self, log: bool = False) -> float:
        """Compute the Bayesian network's probability for the current state."""
        res = super().comp_bayes_net_prob(log=log)

        for edge in self.inc:
            if log:
                res += edge.comp_bayes_net_prob(log=True)
            else:
                res *= edge.comp_bayes_net_prob(log=False)

        return res

    # note: here we gained extra computation time. the former version used to save
    # transition probabilities that were computed before with an @lru_cache decorator
    # since the computation is not done in node anymore, this will not work now.
    # thus we will need to implement a function that checks and caches results
    def comp_trans_prob(self, new_state: int, log: bool = False) -> float:
        """Compute the hidden Markov model's transition probability to a new state."""
        stay_prob = super().comp_trans_prob(new_state, log)
        if new_state == self.state == self.allowed_states[-1]:
            return stay_prob
        if new_state - self.state > 1:
            return -np.inf if log else 0

        for edge in self.inc:
            stay_prob *= edge.comp_stay_prob()
        if self.state == new_state:
            return log(stay_prob) if log else stay_prob
        return log(1-stay_prob) if log else 1-stay_prob
