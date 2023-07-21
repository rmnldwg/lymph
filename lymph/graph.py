"""
Module defining the nodes and edges of the graph representing the lymphatic system.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Union
from functools import wraps
import warnings

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


    def comp_bayes_net_prob(self, log: bool = False) -> float:
        """Compute the Bayesian network's probability for the current state."""
        return 0. if log else 1.


    def comp_trans_prob(self, new_state: int) -> float:
        """Compute the hidden Markov model's transition probability to a new state."""
        if new_state not in self.allowed_states:
            raise ValueError("New state must be one of the allowed states")

        return 0. if new_state < self.state else 1.


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
        self.inc: List[LymphNodeLevel] = []


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


    def comp_trans_prob(self, new_state: int) -> float:
        """Compute the hidden Markov model's transition probability to a `new_state`."""
        trans_prob = super().comp_trans_prob(new_state)

        for edge in self.inc:
            trans_prob *= edge.transition_tensor[
                edge.start.state,
                edge.end.state,
                new_state,
            ]

        return trans_prob


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

        if self.end.is_trinary:
            self.macro_mod = micro_mod

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


    def get_macro_mod(self) -> float:
        """Return the spread probability."""
        if not hasattr(self, "_macro_mod") or self.end.is_binary:
            self._macro_mod = 1.
        return self._macro_mod

    @delete_transition_tensor
    def set_macro_mod(self, new_macro_mod: float) -> None:
        """Set the spread modifier for LNLs with microscopic involvement."""
        if self.end.is_binary:
            warnings.warn("Microscopic spread modifier is not used for binary nodes!")

        if not 0. <= new_macro_mod <= 1.:
            raise ValueError("Microscopic spread modifier must be between 0 and 1!")

        self._macro_mod = new_macro_mod

    macro_mod = property(
        fget=get_macro_mod,
        fset=set_macro_mod,
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


    def comp_bayes_net_prob(self, log: bool = False) -> float:
        """Compute the conditional probability of this edge's child node's state.

        This function dynamically computes the conditional probability that the child
        node is in its state, given the parent node's state and the parameters of the
        edge.
        """
        # TODO: Implement this function
        raise NotImplementedError("Not implemented yet!")


    def comp_transition_tensor(self) -> np.ndarray:
        """Compute the transition factors of the edge.

        The returned array is of shape (s,e,e), where s is the number of states of the
        start node and e is the number of states of the end node.

        Essentially, the tensors computed here contain most of the parametrization of
        the model. They are used to compute the transition matrix.
        """
        num_start = len(self.start.allowed_states)
        num_end = len(self.end.allowed_states)
        tensor = np.ones(shape=(num_start, num_end, num_end))

        for i, start_state in enumerate(self.start.allowed_states):
            if start_state == 2:
                # here we implement how the macroscopic state changes the spread
                spread_prob = self.spread_prob * self.macro_mod
            else:
                spread_prob = self.spread_prob

            for j, end_state in enumerate(self.end.allowed_states):
                # in the growth case, s must be equal to e
                if self.is_growth and start_state != end_state:
                    continue

                for k, new_state in enumerate(self.end.allowed_states):
                    if self.is_growth and end_state == 1:
                        tensor[i,j,k] = (
                            (1. - spread_prob) ** (new_state == 1)
                            * spread_prob ** (new_state == 2)
                        )

                    else:
                        tensor[i,j,k] = (
                            (1. - spread_prob) ** (new_state == 0)
                            * spread_prob ** (new_state == 1)
                        ) ** (end_state == 0 and start_state != 0)

            tensor[i] = np.triu(tensor[i])

        # making the tensor upper triangular, because self-healing is not allowed
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
                return 1 - self.spread_prob * self.macro_mod
            elif self.end.state == 1:
                if self.is_growth:
                    return 1 - self.spread_prob

        if self.start.state == 2:
            if self.end.state == 0:
                return 1 - self.spread_prob

        return 1
