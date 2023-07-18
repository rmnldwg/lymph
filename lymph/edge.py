"""
This module implements the edges of the graph representation of the lymphatic system.
"""
from __future__ import annotations

from typing import Union
import numpy as np

from lymph.node import AbstractNode, LymphNodeLevel, Tumor


class Edge:
    """
    This class represents an arc in the graph representation of the lymphatic system.
    """
    def __init__(
        self,
        start: Union[Tumor, LymphNodeLevel],
        end: LymphNodeLevel,
        spread_prob: float = 0.,
        micro_mod: float = 1.,
    ):
        """Create a new edge between two nodes.

        The `start` node must be a `Tumor` or a `LymphNodeLevel`, and the `end` node
        must be a `LymphNodeLevel`.

        The `spread_prob` parameter is the probability of a tumor or involved LNL to
        spread to the next LNL. The `micro_mod` parameter is a modifier for the spread
        probability in case of only a microscopic node involvement.
        """
        self.start = start
        self.end = end

        self.micro_mod = micro_mod
        self.spread_prob = spread_prob


    def __str__(self):
        """Print basic info."""
        return f"{self.start}-->{self.end}"


    @property
    def start(self) -> Union[Tumor, LymphNodeLevel]:
        """Return the start (parent) node of the edge."""
        return self._start

    @start.setter
    def start(self, new_start: Union[Tumor, LymphNodeLevel]) -> None:
        """Set the start (parent) node of the edge."""
        if not issubclass(new_start.__class__, AbstractNode):
            raise TypeError("Start must be instance of Node!")

        self._start = new_start
        self.start.out.append(self)


    @property
    def end(self) -> LymphNodeLevel:
        """Return the end (child) node of the edge."""
        return self._end

    @end.setter
    def end(self, new_end: LymphNodeLevel) -> None:
        """Set the end (child) node of the edge."""
        if not isinstance(new_end, LymphNodeLevel):
            raise TypeError("End must be instance of Node!")

        self._end = new_end
        self.end.inc.append(self)


    @property
    def name(self) -> str:
        """Return the name of the edge.
        
        This is used to identify it and assign spread probabilities to it in
        the `Unilateral` class.
        """
        return self.start.name + '_to_' + self.end.name


    @property
    def is_growth(self) -> bool:
        """Check if this edge represents a node's growth."""
        return self.start == self.end


    @property
    def is_tumor_spread(self) -> bool:
        """Check if this edge represents spread from a tumor to an LNL."""
        return isinstance(self.start, Tumor)


    def get_micro_mod(self) -> float:
        """Return the spread probability."""
        if not hasattr(self, "_micro_mod"):
            self._micro_mod = 1.
        return self._micro_mod

    def set_micro_mod(self, new_micro_mod: float) -> None:
        """Set the spread modifier for LNLs with microscopic involvement."""
        if not 0. <= new_micro_mod <= 1.:
            raise ValueError("Microscopic spread modifier must be between 0 and 1!")
        self._micro_mod = new_micro_mod

        if hasattr(self, "_trans_factor_matrix"):
            del self._trans_factor_matrix

    micro_mod = property(
        get_micro_mod,
        set_micro_mod,
        doc="Parameter modifying spread probability in case of microscopic involvement",
    )


    def get_spread_prob(self) -> float:
        """Return the spread probability."""
        if not hasattr(self, "_spread_prob"):
            self._spread_prob = 0.
        return self._spread_prob

    def set_spread_prob(self, new_spread_prob):
        """Set the spread probability of the edge."""
        if not 0. <= new_spread_prob <= 1.:
            raise ValueError("Spread probability must be between 0 and 1!")
        self._spread_prob = new_spread_prob

        if hasattr(self, "_trans_factor_matrix"):
            del self._trans_factor_matrix

    spread_prob = property(
        get_spread_prob,
        set_spread_prob,
        doc="Spread probability of the edge",
    )


    def comp_bayes_net_prob(self, log: bool = False) -> float:
        """Compute the conditional probability of this edge's child node's state.

        This function dynamically computes the conditional probability that the child
        node is in its state, given the parent node's state and the parameters of the
        edge.
        """

        # Implementing this is not a priority, but it would be nice to have.
        raise NotImplementedError("Not implemented yet!")


    def _gen_trans_factor_matrix(self) -> None:
        """Generate the transition factor matrix for the edge.

        This matrix has one row for each possible state of the starting `Node` instance,
        and one column for each possible state of the ending `Node` instance.

        This matrix needs to be recomputed every time the parameters of the edge change.
        """
        if self.start.is_binary:
            self._trans_factor_matrix = np.array([
                [1.                   ,               1.],
                [1. - self.spread_prob, self.spread_prob],
            ])

        elif self.start.is_trinary and self.is_growth:
            growth_prob = self.spread_prob
            self._trans_factor_matrix = np.array([
                [1., 1.              , 0.         ],
                [0., 1. - growth_prob, growth_prob],
                [0., 0.              , 1.         ],
            ])

        elif self.start.is_trinary:
            micro_spread_prob = self.micro_mod * self.spread_prob
            self._trans_factor_matrix = np.array([
                [1.                    , 1.               , 1.],
                [1. - micro_spread_prob, micro_spread_prob, 1.],
                [1. - self.spread_prob , self.spread_prob , 1.],
            ])

        else:
            raise NotImplementedError("Only binary and trinary nodes are supported!")


    @property
    def trans_factors(self) -> np.ndarray:
        """Compute the transition factors of the edge.

        These factors are returned in an array with one element for each possible state
        of the `Node` instance at the end of the edge. It can be multiplied with the
        probability of the end node's intrinsic probability to change from its current
        to another state.
        """
        if not hasattr(self, "_trans_factor_matrix"):
            self._gen_trans_factor_matrix()

        return self._trans_factor_matrix[self.start.state]


    def comp_stay_prob(self) -> float:
        """Compute the probability of spread per time-step in the hidden Markov model.

        This function dynamically computes the probability of no spread per time-step,
        i.e. that the child node will stay in the same state given the states of its
        parent nodes, and the parameters of the edge.
        """
        # TODO: I think there's still something missing here: What if the start state
        # and the end state are both 0? Then it should not return 1 - spread_prob.
        if self.end.is_binary:
            if self.end.state == 0:
                return 1 - self.spread_prob

        if self.start.state == 1:
            if self.end.state == 0:
                return 1 - self.spread_prob * self.micro_mod
            elif self.end.state == 1:
                if self.is_growth:
                    return 1 - self.spread_prob

        if self.start.state == 2:
            if self.end.state == 0:
                return 1 - self.spread_prob

        return 1
