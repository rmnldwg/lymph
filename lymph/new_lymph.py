import numpy as np
from numpy.linalg import matrix_power as mat_pow
import scipy as sp 
import scipy.stats
import pandas as pd
import warnings
# from functools import cache
from typing import Union, Optional, List, Dict, Any
import igraph as ig


def change_base(
    number: int, 
    base: int, 
    reverse: bool = False, 
    length: Optional[int] = None
) -> str:
    """Convert an integer into another base.
    
    Args:
        number: Number to convert
        base: Base of the resulting converted number
        reverse: If true, the converted number will be printed in reverse order.
        length: Length of the returned string. If longer than would be 
            necessary, the output will be padded.

    Returns:
        The (padded) string of the converted number.
    """
    
    if base > 16:
        raise ValueError("Base must be 16 or smaller!")
        
    convertString = "0123456789ABCDEF"
    result = ''
    while number >= base:
        result = result + convertString[number % base]
        number = number//base
    if number > 0:
        result = result + convertString[number]
        
    if length is None:
        length = len(result)
    elif length < len(result):
        length = len(result)
        warnings.warn("Length cannot be shorter than converted number.")
        
    pad = '0' * (length - len(result))
        
    if reverse:
        return result + pad
    else:
        return pad + result[::-1]


class System(ig.Graph):
    """Class that models metastatic progression in a lymphatic system by 
    representing it as a directed graph. The progression itself can be modelled 
    via hidden Markov models (HMM) or Bayesian networks (BN).
    """
    
    def __init__(self, graph: dict, *args, **kwargs):
        """Constructor"""
        kwargs["directed"] = True
        num_nodes = len(graph)
        super().__init__(num_nodes, *args, **kwargs)
        
        self.vs["name"]  = [key[1] for key in graph.keys()]
        self.vs["type"]  = [key[0] for key in graph.keys()]
        for source, targets in graph.items():
            self.add_edges([(source[1], target) for target in targets])


    @property
    def state(self) -> List[bool]:
        """Return the currently set state of the system."""
        return self.vs["state"]

    @state.setter
    def state(self, newstate: List[bool]):
        """Sets the state of the system to ``newstate``."""
        self.vs["state"] = newstate
            

    @property
    def spread_probs(self) -> List[float]:
        """Return the spread probabilities of the edges in the network in the 
        order they appear in the graph. If none have been set yet, warn and 
        output random vector, so one can see the shape of the parameter array.
        """
        try:
            return np.array(self.es["weight"])
        except KeyError:
            warnings.warn(
                "Spread probabilities not yet set, returning random values"
            )
            return np.random.uniform(size=len(self.es))
    
    @spread_probs.setter
    def spread_probs(self, new_spread_probs: List[float]):
        """Set the spread probabilities of the edges in the the network in the 
        order they were created from the graph and recompute the transition 
        matrix."""
        self.es["weight"] = new_spread_probs
        self._gen_transition_matrix()
        self._node_transition_prob.cache_clear()
    
    
    def _gen_state_list(self):
        """Generate list of all possible hidden states. The tumor(s) are always 
        in the positive/involved state."""
        num_tumors = len(self.vs.select(type="tumor"))
        num_lnls   = len(self.vs.select(type="lnl"))
        self._state_list = np.zeros(
            shape=(2**num_lnls, num_lnls + num_tumors), 
            dtype=int
        )
        for i in range(2**num_lnls):
            tmp = [int(digit) for digit in change_base(i, 2, length=num_lnls)]
            state = np.concatenate([[True] * num_tumors, tmp])
            self._state_list[i] = state
    
    @property
    def state_list(self):
        """Return list of all possible hidden states. They are arranged in the 
        same order as the nodes in the network/graph. The first nodes 
        representing the tumor are alwaus ``True``."""
        try:
            return self._state_list
        except AttributeError:
            self._gen_state_list()
            return self._state_list
    
    
    def _gen_mask(self):
        """Generate the index mask."""
        self._mask = {}
        for i,start_state in enumerate(self.state_list):
            self._mask[i] = []
            for j,end_state in enumerate(self.state_list):
                if not np.any(np.greater(start_state, end_state)):
                    self._mask[i].append(j)
    
    @property
    def mask(self):
        """Return a dictionary with keys for each possible hidden state. The 
        respective value is then a list of all hidden state indices that can be 
        reached from that key's state. This allows the model to skip the 
        expensive computation of entries in the transition matrix that are zero 
        anyways, because self-healing is forbidden.
        
        For example: The hidden state ``[True, True, False]`` in a network 
        with only one tumor and two LNLs (one involved, one healthy) corresponds 
        to the index ``1`` and can only evolve into the state 
        ``[True, True, True]``, which has index 2. So, the key-value pair for 
        that particular hidden state would be ``1: [2]``.
        """
        try:
            return self._mask
        except AttributeError:
            self._gen_mask()
            return self._mask


    def _node_transition_prob(in_weights: List[float], in_states: List[int]):
        """Compute the probability for a node to remain healthy or become 
        involved based solely on the weights of that node's incoming arcs and 
        those arc's source node's state.
        """
        stay_prob = 1.
        for weight, state in zip(in_weights, in_states):
            stay_prob *= (1 - weight) ** state
        
        return [stay_prob, 1 - stay_prob]    
    
    def _gen_transition_matrix(self):
        """Generate transition matrix.
        
        The computation probably doesn't look very intuitive, so here's an 
        attempt at explaining: 
        
        1. The adjacency matrix of the graph is element-wise multiplied with 
        every possible hidden state, thereby only "activating" those LNLs that 
        are actually cancerous.
        2. The resulting (3D) array is user to compute the probability that a 
        node level does not become involved, which is essentially the product 
        of one minus every entry.
        3. We end up with an array of probabilities for every starting state 
        that contains the probability for every LNL remain healthy. A 
        complementary matrix is computed that computes one minus that value to 
        get the value for every LNL to become involved.
        4. Finally, for every possible starting state access the probabilities 
        for every LNL to remain healthy or become involved, depending on the 
        respective end state. Overwrite this for thos LNLs that are already 
        involved in the starting state (they stay in the sick state with 
        probability 1).
        
        Steps 1 to 3 are computed once using lots of ``numpy`` matrix magic. 
        Step 4 is then computed for every starting state and for every end 
        state that has a non-zero probability (no self-healing).
        """
        num_lnls = len(self.vs.select(type="lnl"))
        node_indexer = range(len(self.vs))
        self._transition_matrix = np.zeros(
            shape=(2**num_lnls, 2**num_lnls), dtype=float
        )
        adj_mat = np.array(self.get_adjacency(attribute="weight").data)
        state_active_adj_arr = np.einsum("ij,jk->jik", self.state_list, adj_mat)
        no_spread_probs = np.prod(1 - state_active_adj_arr, axis=0)
        total_probs = np.stack(
            [no_spread_probs, 1 - no_spread_probs], axis=-1
        )
        for i,start_state in enumerate(self.state_list):
            for j in self.mask[i]:
                end_state = self.state_list[j]
                lnl_probs = total_probs[i, node_indexer, end_state]
                lnl_probs = np.maximum(lnl_probs, start_state)
                self._transition_matrix[i,j] = np.prod(lnl_probs)
    
    def _gen_transition_matrix_var(self):
        """"""
        num_lnl = len(self.vs.select(type="lnl"))
        self._transition_matrix = np.zeros(
            shape=(2**num_lnl, 2**num_lnl), dtype=float
        )
        for i,start_state in enumerate(self.state_list):
            self.vs["state"] = start_state
            idx = start_state == 0
            for j in self.mask[i]:
                end_state = self.state_list[j][idx]
                healthy_nodes = self.vs[idx]
                state_transition_prob = 1.
                for k,node in enumerate(healthy_nodes):
                    
                    state_transition_prob *= self._node_transition_prob(
                        node, start_state
                    )
    
    @property
    def transition_matrix(self):
        """Return the transition matrix :math:`\\mathbf{A}`, which contains the 
        probability to transition from any state to any other state within one 
        time-step :math:`P \\left( X_{t+1} \\mid X_t \\right)`. 
        :math:`\\mathbf{A}` is a square matrix with size ``(# of states)``. The 
        lower diagonal is zero, because self-healing is forbidden.
        """
        try:
            return self._transition_matrix
        except AttributeError:
            self._gen_transition_matrix()
            return self._transition_matrix
        