import numpy as np
import warnings
from functools import lru_cache

from typing import Tuple


@lru_cache
def node_trans_prob(in_states: Tuple[int], in_weights: Tuple[float]):
    """Compute probability of a random variable to remain in its state (0) or 
    switch to be involved (1) based on its parent's states and the weights of 
    the connecting arcs. Cached for better performance.
    
    Args:
        in_states: States of the parent nodes.
        in_weights: Weights of the incoming arcs.
    
    Returns:
        Probability to remain healthy and probability to become metastatic 
        as a list of length 2.
    
    Note:
        This function should only be called when the :class:`Node`, to which the 
        incoming weights and states refer to, is in state ``0``. Otherwise, 
        meaning it is already involved/metastatic, it must stay in this state 
        no matter what, because self-healing is forbidden.
    """
    stay_prob = 1.
    for state, weight in zip(in_states, in_weights):
        stay_prob *= (1. - weight) ** state
    return [stay_prob, 1. - stay_prob]


class Node(object):
    """Very basic class for tumors and lymph node levels (LNLs) in a lymphatic 
    system. This serves as one part of a lightweight network setup (the other 
    one is the :class:`Edge` class).
    """
    def __init__(self, name: str, state: int = 0, typ: str = "lnl"):
        """
        Args:
            name: Name of the node.
            state: Current state this LNL is in. Can be in {0, 1}.
            typ: Can be either ``"lnl"``, ``"tumor"``.
        """
        self.name = name
        self.typ = typ
        self.state = state

        self.inc = []
        self.out = []
    
    
    def __str__(self):
        """Print basic info"""
        num_inc = len(self.inc)
        num_out = len(self.out)
        return f"{num_inc:d} --> {self.name} ({self.typ}) --> {num_out:d}"
    
    
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
            self._state = newstate
        elif self.typ == "tumor":
            self._state = 1
    
    
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
        if newtyp in ["tumor", "lnl"]:
            self._typ = newtyp
        else:
            raise ValueError("Only types 'tumor' and 'lnl' are available.")


    def trans_prob(self) -> float:
        """Compute the transition probabilities from the current state to all
        other possible states (which is only two).
        """
        stay_prob = 1.

        if self.state:
            return [1., 0.]

        for edge in self.inc:
            stay_prob *= (1 - edge.t) ** edge.start.state
        
        return [stay_prob, 1. - stay_prob]


    def obs_prob(
        self, obs: int, obstable: np.ndarray = np.eye(2)) -> float:
        """Compute the probability of observing a certain diagnose, given its 
        current state.

        Args:
            obs: Diagnose/observation for the node.
            obstable: 2x2 matrix containing info about sensitivity and 
                specificty of the observational/diagnostic modality from which 
                `obs` was obtained.

        Returns:
            The probability of observing the given diagnose.
        """
        return obstable[obs, self.state]


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
