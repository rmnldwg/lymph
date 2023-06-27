"""
This module implements the edges of the graph representation of the lymphatic system.
"""
from __future__ import annotations

from typing import Union

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
        self.name = self.start.name + '_to_' + self.end.name

    # here I would add the spread_probability
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
    def is_growth(self) -> bool:
        """Check if this edge represents a node's growth."""
        return self.start == self.end


    @property
    def is_tumor_spread(self) -> bool:
        """Check if this edge represents spread from a tumor to an LNL."""
        return isinstance(self.start, Tumor)


    def set_micro_mod(self, new_micro_mod: float) -> None:
        """Set the spread modifier for LNLs with microscopic involvement."""
        new_micro_mod = float(new_micro_mod)

        if not (0. <= new_micro_mod <= 1.):
            raise ValueError("Microscopic spread modifier must be between 0 and 1!")

        self.micro_mod = new_micro_mod
    
    def set_spread_prob(self, new_spread_prob: float) -> None:
        """Set the spread probability of the edge."""
        new_spread_prob = float(new_spread_prob)

        if not (0. <= new_spread_prob <= 1.):
            raise ValueError("Spread probability must be between 0 and 1!")

        self.spread_prob = new_spread_prob


    def comp_bayes_net_prob(self, log: bool = False) -> float:
        """Compute the conditional probability of this edge's child node's state.

        This function dynamically computes the conditional probability that the child
        node is in its state, given the parent node's state and the parameters of the
        edge.
        """

        # Implementing this is not a priority, but it would be nice to have.
        raise NotImplementedError("Not implemented yet!")


    def comp_stay_prob(self) -> float:
        """Compute the probability of spread per time-step in the hidden Markov model.

        This function dynamically computes the probability of no spread per time-step i.e. 
        that the child node will stay in the same state given the states of its parent nodes,
        and the parameters of the edge.
        """
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


