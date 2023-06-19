from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np


class Node(object):
    """Very basic class for tumors and lymph node levels (LNLs) in a lymphatic
    system. This serves as one part of a lightweight network setup (the other
    one is the :class:`Edge` class).
    """
    def __init__(self, name: str, state: int = 0, typ: str = "lnl", allowed_states: int = 3):
        """
        Args:
            name: Name of the node.
            state: Current state this LNL is in. Can be in any allowed state e.g. for 3: {0, 1, 2}.
            typ: Can be either ``"lnl"``, ``"tumor"``.
        """
        self.allowed_states = int(allowed_states)
        if type(name) is not str:
            raise TypeError("Name of node must be a string")
        if int(state) not in list(range(allowed_states)):
            raise ValueError("State must be castable to the allowed states")
        if typ not in ["lnl", "tumor"]:
            raise ValueError("Typ of node must be either `lnl` or `tumor`")

        self.name = name
        self.typ = typ
        self.state = int(state)
        self.inc = []
        self.out = []


    def __str__(self):
        """Print basic info"""
        return self.name


    @property
    def state(self) -> int:
        """Return the state of the node."""
        try:
            return self._state
        except AttributeError:
            raise AttributeError("State has not been set yet.")
        

    @state.setter
    def state(self, newstate: int):
        """Set the state of the node and make sure the state of a tumor node
        cannot be changed."""
        if self.typ == "lnl":
            if int(newstate) not in list(range(self.allowed_states)):
                raise ValueError("State of node must be either 0, 1,... allowed_states - 1")
            self._state = int(newstate)

        elif self.typ == "tumor":
            self._state = self.allowed_states - 1


    @property
    def typ(self) -> str:
        """Return the type of the node, which can be ``"tumor"`` or ``"lnl"``.
        """
        try:
            return self._typ
        except AttributeError:
            raise AttributeError("Type of node has not been set yet.")
        

    @typ.setter
    def typ(self, newtyp: str):
        """Set the type of the node (either ``"tumor"`` or ``"lnl"``)."""
        if newtyp == "tumor":
            self._typ = newtyp
            self.state = self.allowed_states - 1
        elif newtyp == "lnl":
            self._typ = newtyp
        else:
            raise ValueError("Only types 'tumor' and 'lnl' are available.")
        

    # caching does not work anymore here
    def trans_prob(self, new_state) -> float:
        """Compute probability of a random variable to transition into new_state, 
        cached method for better performance.

        Args:
            new_state: new state of node

        Returns:
            Probability to transition to the new_state
        """
        if new_state < self.state:
            return 0
        else:
            healthy_prob = 1
            growth_prob = 1
            for edge in self.inc:
                if edge.is_growth:
                    growth_prob = edge.t
                else:
                    healthy_prob *= (1. - edge.t)

            transition_list = [healthy_prob, 1 - healthy_prob, 1 - growth_prob, growth_prob]
            #in theory we do not need to calculate the whole list. we could do some optimizations here, but I think we should go in this direction of using a function that only takes new state as input.
            if new_state == 0:
                return transition_list[0]
            elif new_state == 1 and self.state == 0:
                return transition_list[1]
            elif new_state == 1 and self.state == 1:
                return transition_list[2] if self.allowed_states == 3 else 1
            elif new_state == 2 and self.state == 1:
                return transition_list[3]
            elif new_state == 2 and self.state == 2:
                return 1
            elif new_state == 2 and self.state == 0:
                return 0


    def obs_prob(
        self, obs: Union[float, int], obstable: np.ndarray = np.eye(2)
    ) -> float:
        """Compute the probability of observing a certain diagnose, given its
        current state. If the diagnose is unspecified (e.g. ``None`` or
        ``NaN``), the probability is of that "diagnose" is 1.

        Args:
            obs: Diagnose/observation for the node.
            obstable: 2xallowed_states matrix containing info about sensitivity and
                specificty of the observational/diagnostic modality from which
                `obs` was obtained.

        Returns:
            The probability of observing the given diagnose.
        """
        if obs is None or np.isnan(obs):
            return 1.
        else:
            return obstable[int(obs), self.state]


    def bn_prob(self, log: bool = False) -> float:
        """Computes the conditional probability of a node being in the state it
        is in, given its parents are in the states they are in.

        Args:
            log: If ``True``, returns the log-probability.
                (default: ``False``)

        Returns:
            The conditional (log-)probability.
        """
        res = 1.
        for edge in self.inc:
            res *= (1 - edge.t)**edge.start.state

        res *= (-1)**self.state
        res += self.state

        return np.log(res) if log else res
