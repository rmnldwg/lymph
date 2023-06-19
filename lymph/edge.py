from __future__ import annotations

from .node import Node


class Edge(object):
    """Minimalistic class for the connections between lymph node levels (LNLs)
    represented by the :class:`Node` class. It only holds its start and end
    node, as well as the transition probability.
    """
    def __init__(self, start: Node, end: Node, base_t: float = 0., microscopic_parameter: float = 1.):
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


        self.start = start
        self.start.out.append(self)
        self.end = end
        self.end.inc.append(self)
        self.microscopic_parameter = float(microscopic_parameter)
        self.base_t = base_t
        
        
    def __str__(self):
        """Print basic info"""
        return f"{self.start}-{100 * self.t:.1f}%->{self.end}"
    
    
    @property
    def microscopic_parameter(self):
        """Return the microscopic spread parameter that is multiplied with the
        base transition probabilities in case of a microscopic spread
        """
        return self._microscopic_parameter
    
    @microscopic_parameter.setter
    def microscopic_parameter(self, new_microscopic_parameter: float):
        """Set the microscopic spread parameter for LNLs
        """
        if new_microscopic_parameter <= 1. and new_microscopic_parameter >= 0.:
            self._microscopic_parameter = new_microscopic_parameter
            if hasattr(self, "_t"):
                del self._t
        else:
            raise ValueError("microscopic spread parameter must be between 0 and 1")
        
        
    @property
    def is_growth(self) -> bool:
        """Check if this edge represents a node's growth."""
        return self.start == self.end
    
    
    @property
    def base_t(self):
        """Holds the base transition probability for transition to the next state.
        For growth edges the base transition probability is the growth probability.
        For edges between LNLs the base transition probability holds the transition
        probability for a macroscopic parent node/a involved node in the binary model.
        The base_t is multiplied with the macroscopic parameter if needed.      
        """
        return self._base_t

    @base_t.setter
    def base_t(self, new_base_t: float):
        """ Sets the base_t parameter and the transition _t parameter.
        This method checks the edge type and parent node to compute the final 
        transition probability.
        """
        self._base_t = new_base_t 
        if self.start.state == 0:
            self._t = 0.
        elif self.is_growth:
            self._t = self._base_t
        elif new_base_t <= 1. and new_base_t >= 0.:
            self._t = self._base_t * self.microscopic_parameter if self.start.state == 1 and self.start.allowed_states == 3 else self._base_t
        else:
            raise ValueError("Transmission probability must be between 0 and 1")
        

    @property
    def t(self):
        """Returns the transition probability applied by an edge after considering
        whether it is a growth/microscopic/macroscopic edge.
        """
        try:
            return self._t
        except AttributeError:
            self.base_t = self._base_t
            return self._t