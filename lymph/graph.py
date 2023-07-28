"""
Module defining the nodes and edges of the graph representing the lymphatic system.
"""
from __future__ import annotations

import warnings
from functools import wraps
from typing import Callable

import numpy as np


class AbstractNode:
    """
    Abstract base class for nodes in the graph reprsenting the lymphatic system.
    """
    def __init__(
        self,
        name: str,
        state: int,
        allowed_states: list[int] | None = None,
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

    def __repr__(self) -> str:
        """Return a string representation of the node."""
        cls_name = type(self).__name__
        return (
            f"{cls_name}("
            f"name={self.name!r}, "
            f"state={self.state!r}, "
            f"allowed_states={self.allowed_states!r})"
        )


    @property
    def name(self) -> str:
        """Return the name of the node."""
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """Set the name of the node."""
        self._name = str(new_name)


    @property
    def state(self) -> int:
        """Return the state of the node."""
        return self._state

    @state.setter
    def state(self, new_state: int) -> None:
        """Set the state of the node."""
        new_state = int(new_state)

        if new_state not in self.allowed_states:
            raise ValueError("State of node must be one of the allowed states")

        self._state = new_state


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
    def __init__(self, name: str, state: int = 1) -> None:
        """Create a new tumor.

        A tumor can only ever be in one state, and it cannot change its state.
        """
        allowed_states = [state]
        super().__init__(name, state, allowed_states)


    def __str__(self):
        """Print basic info"""
        return f"Tumor '{super().__str__()}'"


class LymphNodeLevel(AbstractNode):
    """A lymph node level (LNL) in the graph representation of the lymphatic system."""
    def __init__(
        self,
        name: str,
        state: int = 0,
        allowed_states: list[int] | None = None,
    ) -> None:
        """Create a new lymph node level."""

        super().__init__(name, state, allowed_states)

        # LNLs can also have incoming edge connections
        self.inc: list[LymphNodeLevel] = []


    @classmethod
    def binary(cls, name: str, state: int = 0) -> LymphNodeLevel:
        """Create a new binary LNL."""
        return cls(name, state, [0, 1])

    @classmethod
    def trinary(cls, name: str, state: int = 0) -> LymphNodeLevel:
        """Create a new trinary LNL."""
        return cls(name, state, [0, 1, 2])


    def __str__(self):
        """Print basic info"""
        narity = "binary" if self.is_binary else "trinary"
        return f"{narity} LNL '{super().__str__()}'"


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
        res = 0 if log else 1

        for edge in self.inc:
            if log:
                res += edge.comp_bayes_net_prob(log=True)
            else:
                res *= edge.comp_bayes_net_prob(log=False)

        return res


    def comp_trans_prob(self, new_state: int) -> float:
        """Compute the hidden Markov model's transition probability to a `new_state`."""
        if new_state == self.state:
            stay_prob = 1.
            for edge in self.inc:
                edge_prob = edge.transition_tensor[edge.parent.state, self.state, new_state]
                stay_prob *= edge_prob
            return stay_prob

        transition_prob = 0.
        for edge in self.inc:
            edge_prob = edge.transition_tensor[edge.parent.state, self.state, new_state]
            transition_prob = 1. - (1. - transition_prob) * (1. - edge_prob)


def delete_transition_tensor(setter: Callable) -> Callable:
    """Decorator to delete the transition tensor of the edge.

    This decorator is used to delete the transition tensor of the edge whenever
    the spread probability or the spread modifier is changed.
    """
    @wraps(setter)
    def wrapper(self, *args, **kwargs):
        del self.transition_tensor
        return setter(self, *args, **kwargs)
    return wrapper


class Edge:
    """
    This class represents an arc in the graph representation of the lymphatic system.
    """
    def __init__(
        self,
        parent: Tumor | LymphNodeLevel,
        child: LymphNodeLevel,
        spread_prob: float = 0.,
        micro_mod: float = 1.,
    ):
        """Create a new edge between two nodes.

        The `parent` node must be a `Tumor` or a `LymphNodeLevel`, and the `child` node
        must be a `LymphNodeLevel`.

        The `spread_prob` parameter is the probability of a tumor or involved LNL to
        spread to the next LNL. The `micro_mod` parameter is a modifier for the spread
        probability in case of only a microscopic node involvement.
        """
        self.parent: Tumor | LymphNodeLevel = parent
        self.child: LymphNodeLevel = child

        if self.child.is_trinary:
            self.micro_mod = micro_mod

        self.spread_prob = spread_prob


    def __str__(self) -> str:
        """Print basic info."""
        return f"Edge {self.name.replace('_', ' ')}"

    def __repr__(self) -> str:
        """Print basic info."""
        cls_name = type(self).__name__
        return (
            f"{cls_name}("
            f"parent={self.parent!r}, "
            f"child={self.child!r}, "
            f"spread_prob={self.spread_prob!r}, "
            f"micro_mod={self.micro_mod!r})"
        )


    @property
    def parent(self) -> Tumor | LymphNodeLevel:
        """Return the parent node that drains lymphatically via the edge."""
        return self._parent

    @parent.setter
    def parent(self, new_parent: Tumor | LymphNodeLevel) -> None:
        """Set the parent node of the edge."""
        if not issubclass(new_parent.__class__, AbstractNode):
            raise TypeError("Start must be instance of Node!")

        self._parent = new_parent
        self.parent.out.append(self)


    @property
    def child(self) -> LymphNodeLevel:
        """Return the child node of the edge, receiving lymphatic drainage."""
        return self._child

    @child.setter
    def child(self, new_child: LymphNodeLevel) -> None:
        """Set the end (child) node of the edge."""
        if not isinstance(new_child, LymphNodeLevel):
            raise TypeError("End must be instance of Node!")

        self._child = new_child
        self.child.inc.append(self)


    @property
    def name(self) -> str:
        """Return the name of the edge.

        This is used to identify it and assign spread probabilities to it in
        the `Unilateral` class.
        """
        return self.parent.name + '_to_' + self.child.name


    @property
    def is_growth(self) -> bool:
        """Check if this edge represents a node's growth."""
        return self.parent == self.child


    @property
    def is_tumor_spread(self) -> bool:
        """Check if this edge represents spread from a tumor to an LNL."""
        return isinstance(self.parent, Tumor)


    def get_micro_mod(self) -> float:
        """Return the spread probability."""
        if not hasattr(self, "_micro_mod") or self.child.is_binary:
            self._micro_mod = 1.
        return self._micro_mod

    @delete_transition_tensor
    def set_micro_mod(self, new_micro_mod: float) -> None:
        """Set the spread modifier for LNLs with microscopic involvement."""
        if self.child.is_binary:
            warnings.warn("Microscopic spread modifier is not used for binary nodes!")

        if not 0. <= new_micro_mod <= 1.:
            raise ValueError("Microscopic spread modifier must be between 0 and 1!")

        self._micro_mod = new_micro_mod

    micro_mod = property(
        fget=get_micro_mod,
        fset=set_micro_mod,
        doc="Parameter modifying spread probability in case of macroscopic involvement",
    )


    def get_spread_prob(self) -> float:
        """Return the spread probability."""
        if not hasattr(self, "_spread_prob"):
            self._spread_prob = 0.
        return self._spread_prob

    @delete_transition_tensor
    def set_spread_prob(self, new_spread_prob):
        """Set the spread probability of the edge."""
        if not 0. <= new_spread_prob <= 1.:
            raise ValueError("Spread probability must be between 0 and 1!")
        self._spread_prob = new_spread_prob

    spread_prob = property(
        fget=get_spread_prob,
        fset=set_spread_prob,
        doc="Spread probability of the edge",
    )


    def comp_bayes_prob(self, log: bool = False) -> float:
        """Compute the conditional probability of this edge's child node's state.

        This function dynamically computes the conditional probability that the child
        node is in its state, given the parent node's state and the parameters of the
        edge.
        """
        # TODO: Implement this function
        raise NotImplementedError("Not implemented yet!")


    def comp_transition_tensor(self) -> np.ndarray:
        """Compute the transition factors of the edge.

        The returned array is of shape (p,c,c), where p is the number of states of the
        parent node and c is the number of states of the child node.

        Essentially, the tensors computed here contain most of the parametrization of
        the model. They are used to compute the transition matrix.
        """
        num_parent = len(self.parent.allowed_states)
        num_child = len(self.child.allowed_states)
        tensor = np.stack([np.eye(num_child)] * num_parent)

        # this should allow edges from trinary nodes to binary nodes
        pad = [0.] * (num_child - 2)

        if self.is_tumor_spread:
            # NOTE: Here we define how tumors spread to LNLs
            tensor[0, 0, :] = np.array([1. - self.spread_prob, self.spread_prob, *pad])
            return tensor

        if self.is_growth:
            # In the growth case, we can assume that two things:
            # 1. parent and child state are the same
            # 2. the child node is trinary
            tensor[1, 1, :] = np.array([0., (1 - self.spread_prob), self.spread_prob])
            return tensor

        if self.parent.is_trinary:
            # NOTE: here we define how the micro_mod affects the spread probability
            micro_spread = self.spread_prob * self.micro_mod
            tensor[1,0,:] = np.array([1. - micro_spread, micro_spread, *pad])

            macro_spread = self.spread_prob
            tensor[2,0,:] = np.array([1. - macro_spread, macro_spread, *pad])

            return tensor

        tensor[1,0,:] = np.array([1. - self.spread_prob, self.spread_prob, *pad])
        return tensor


    @property
    def transition_tensor(self) -> np.ndarray:
        """Return the transition tensor of the edge.

        This tensor of the shape (s,e,e) contains the transition probabilities for
        the `Node` at this instance's end to transition from any starting state to
        any new state, given any possible state of the `Node` at the start of this
        edge.

        The correct term can be accessed like this:
        >>> edge.transition_tensor[start_state, end_state, new_state]
        """
        if not hasattr(self, "_transition_tensor"):
            self._transition_tensor = self.comp_transition_tensor()

        return self._transition_tensor

    @transition_tensor.deleter
    def transition_tensor(self) -> None:
        """Delete the transition tensor of the edge."""
        if hasattr(self, "_transition_tensor"):
            del self._transition_tensor
