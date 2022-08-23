from __future__ import annotations

from .node import Node


class Edge(object):
    """Minimalistic class for the connections between lymph node levels (LNLs)
    represented by the :class:`Node` class. It only holds its start and end
    node, as well as the transition probability.
    """
    def __init__(self, start: Node, end: Node, t: float = 0.):
        """
        Args:
            start: Parent node
            end: Child node
            t: Transition probability in case start-Node has state 1 (microscopic
                involvement).
        """
        if not isinstance(start, Node):
            raise TypeError("Start must be instance of Node!")
        if not isinstance(end, Node):
            raise TypeError("End must be instance of Node!")
        if start == end:
            raise ValueError("Start and end node must be different")

        self.start = start
        self.start.out.append(self)
        self.end = end
        self.end.inc.append(self)
        self.t = t


    def __str__(self):
        """Print basic info"""
        return f"{self.start}-{100 * self.t:.1f}%->{self.end}"

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, new_t: float):
        if new_t <= 1. and new_t >= 0.:
            self._t = new_t
        else:
            raise ValueError("Transmission probability must be between 0 and 1")