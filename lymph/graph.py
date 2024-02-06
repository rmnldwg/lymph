"""
Module defining the nodes and edges of the graph representing the lymphatic system.

Anything related to the network of nodes and edges is defined here. This includes the
nodes themselves (either :py:class:`~Tumor` or :py:class:`~LymphNodeLevel`), the edges
(:py:class:`~Edge`), and the graph (:py:class:`~Representation`).

The nodes and edges are used to define the structure of the graph, which may then be
accessed via the :py:class:`~Representation` class. This in turn is then used to
compute e.g. the transition matrix of the model.
"""
from __future__ import annotations

import base64
import warnings
from itertools import product
from typing import Iterable

import numpy as np

from lymph.helper import check_unique_names, comp_transition_tensor, trigger


class AbstractNode:
    """Abstract base class for nodes in the graph reprsenting the lymphatic system."""
    def __init__(
        self,
        name: str,
        state: int,
        allowed_states: list[int] | None = None,
    ) -> None:
        """Make a new node.

        Upon initialization, the ``name`` and ``state`` of the node must be provided.
        The ``state`` must be one of the ``allowed_states``. The constructor makes sure that
        the ``allowed_states`` are a list of ints, even when, e.g., a tuple of floats
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
        self.out: list[Edge] = []


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
        """Compute the probability of the diagnosis ``obs``, given the current state.

        The ``obs_table`` is a 2D array with the rows corresponding to the states and
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
        """Create a new tumor node.

        It can only ever be in one ``state``, which is implemented such that the
        ``allowed_states`` are set to ``[state]``.
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
        self.inc: list[Edge] = []


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
        if self.is_trinary:
            raise NotImplementedError("Trinary nodes are not yet supported")

        res = (-1) ** self.state
        for edge in self.inc:
            parent_state = 0 if isinstance(edge.parent, Tumor) else edge.parent.state
            res *= edge.transition_tensor[parent_state, 0, 0]

        res += self.state
        return np.log(res) if log else res


    def comp_trans_prob(self, new_state: int) -> float:
        """Compute the hidden Markov model's transition probability to a ``new_state``."""
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


class Edge:
    """This class represents an arc in the graph representation of the lymph system."""
    def __init__(
        self,
        parent: Tumor | LymphNodeLevel,
        child: LymphNodeLevel,
        spread_prob: float = 0.,
        micro_mod: float = 1.,
        callbacks: list[callable] | None = None,
    ) -> None:
        """Create a new edge between two nodes.

        The ``parent`` node must be a :py:class:`Tumor` or a :py:class:`LymphNodeLevel`,
        and the ``child`` node must be a :py:class:`LymphNodeLevel`.

        The ``spread_prob`` parameter is the probability of a tumor or involved LNL to
        spread to the next LNL. The ``micro_mod`` parameter is a modifier for the spread
        probability in case of only a microscopic node involvement.
        """
        self.trigger_callbacks = []
        if callbacks is not None:
            self.trigger_callbacks += callbacks

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
    @trigger
    def parent(self, new_parent: Tumor | LymphNodeLevel) -> None:
        """Set the parent node of the edge."""
        if hasattr(self, '_parent'):
            self.parent.out.remove(self)

        if not issubclass(new_parent.__class__, AbstractNode):
            raise TypeError("Start must be instance of Node!")

        self._parent = new_parent
        self.parent.out.append(self)


    @property
    def child(self) -> LymphNodeLevel:
        """Return the child node of the edge, receiving lymphatic drainage."""
        return self._child

    @child.setter
    @trigger
    def child(self, new_child: LymphNodeLevel) -> None:
        """Set the end (child) node of the edge."""
        if hasattr(self, '_child'):
            self.child.inc.remove(self)

        if not isinstance(new_child, LymphNodeLevel):
            raise TypeError("End must be instance of Node!")

        self._child = new_child
        self.child.inc.append(self)


    @property
    def name(self) -> str:
        """Return the name of the edge.

        This is used to identify and assign spread probabilities to it in the
        :py:class:`~models.Unilateral` class.
        """
        if self.is_growth:
            return self.parent.name

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

    @trigger
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

    @trigger
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


    def get_params(
        self,
        param: str | None = None,
        as_dict: bool = False,
    ) -> float | Iterable[float] | dict[str, float]:
        """Return the value of the parameter ``param`` or all params in a dict.

        See Also:
            :py:meth:`lymph.diagnose_times.Distribution.get_params`
            :py:meth:`lymph.diagnose_times.DistributionsUserDict.get_params`
            :py:meth:`lymph.models.Unilateral.get_params`
            :py:meth:`lymph.models.Bilateral.get_params`
        """
        if self.is_growth:
            params = {"growth": self.get_spread_prob()}
            return params if as_dict else params[param]

        params = {"spread": self.get_spread_prob()}
        if self.child.is_trinary and not self.is_tumor_spread:
            params["micro"] = self.get_micro_mod()

        if param is not None:
            return params[param]

        return params if as_dict else params.values()


    def set_params(
        self,
        growth: float | None = None,
        spread: float | None = None,
        micro: float | None = None,
    ) -> None:
        """Set the values of the edge's parameters.

        See Also:
            :py:meth:`lymph.diagnose_times.Distribution.set_params`
            :py:meth:`lymph.diagnose_times.DistributionsUserDict.set_params`
        """
        if self.is_growth:
            return self.set_spread_prob(growth) if growth is not None else None

        if spread is not None:
            self.set_spread_prob(spread)
        if self.child.is_trinary and not self.is_tumor_spread and micro is not None:
            self.set_micro_mod(micro)


    @property
    def transition_tensor(self) -> np.ndarray:
        """Return the transition tensor of the edge.

        See Also:
            :py:function:`lymph.helper.comp_transition_tensor`
        """
        return comp_transition_tensor(
            num_parent=len(self.parent.allowed_states),
            num_child=len(self.child.allowed_states),
            is_tumor_spread=self.is_tumor_spread,
            is_growth=self.is_growth,
            spread_prob=self.spread_prob,
            micro_mod=self.micro_mod,
        )


class Representation:
    """Class holding the graph structure of the model.

    This class allows accessing the connected nodes (:py:class:`Tumor` and
    :py:class:`LymphNodeLevel`) and edges (:py:class:`Edge`) of the :py:mod:`models`.
    """
    def __init__(
        self,
        graph_dict: dict[tuple[str], list[str]],
        tumor_state: int | None = None,
        allowed_states: list[int] | None = None,
        on_edge_change: list[callable] | None = None,
    ) -> None:
        """Create a new graph representation of nodes and edges.

        The ``graph_dict`` is a dictionary that defines which nodes are created and
        with what edges they are connected. The keys of the dictionary are tuples of
        the form ``(node_type, node_name)``. The ``node_type`` can be either ``"tumor"``
        or ``"lnl"``. The ``node_name`` is a string that uniquely identifies the node.
        The values of the dictionary are lists of node names to which the key node
        should be connected.
        """
        if allowed_states is None:
            allowed_states = [0, 1]

        if tumor_state is None:
            tumor_state = allowed_states[-1]

        check_unique_names(graph_dict)
        self._init_nodes(graph_dict, tumor_state, allowed_states)
        self._init_edges(graph_dict, on_edge_change)


    def _init_nodes(self, graph, tumor_state, allowed_lnl_states):
        """Initialize the nodes of the graph."""
        self._nodes: dict[str, Tumor | LymphNodeLevel] = {}

        for node_type, node_name in graph:
            if node_type == "tumor":
                tumor = Tumor(name=node_name, state=tumor_state)
                self._nodes[node_name] = tumor
            elif node_type == "lnl":
                lnl = LymphNodeLevel(name=node_name, allowed_states=allowed_lnl_states)
                self._nodes[node_name] = lnl


    @property
    def nodes(self) -> dict[str, Tumor | LymphNodeLevel]:
        """List of both :py:class:`~Tumor` and :py:class:`~LymphNodeLevel` instances."""
        return self._nodes

    @property
    def tumors(self) -> dict[str, Tumor]:
        """List of all :py:class:`~Tumor` nodes in the graph."""
        return {n: t for n, t in self.nodes.items() if isinstance(t, Tumor)}

    @property
    def lnls(self) -> dict[str, LymphNodeLevel]:
        """List of all :py:class:`~LymphNodeLevel` nodes in the graph."""
        return {n: l for n, l in self.nodes.items() if isinstance(l, LymphNodeLevel)}


    @property
    def allowed_states(self) -> list[int]:
        """Return the list of allowed states for each :py:class:`~LymphNodeLevel`."""
        return next(iter(self.lnls.values())).allowed_states

    @property
    def is_binary(self) -> bool:
        """Indicate if the model is binary.

        Returns ``True`` if all :py:class:`~LymphNodeLevel` instances are binary,
        ``False`` otherwise.
        """
        res = {node.is_binary for node in self.lnls.values()}

        if len(res) != 1:
            raise RuntimeError("Not all lnls have the same number of states")

        return res.pop()

    @property
    def is_trinary(self) -> bool:
        """Returns ``True`` if the graph is trinary, ``False`` otherwise.

        Similar to :py:meth:`~Unilateral.is_binary`."""
        res = {node.is_trinary for node in self.lnls.values()}

        if len(res) != 1:
            raise RuntimeError("Not all lnls have the same number of states")

        return res.pop()


    def _init_edges(
        self,
        graph: dict[tuple[str, str], list[str]],
        on_edge_change: list[callable]
    ) -> None:
        """Initialize the edges of the ``graph``.

        Every one of the provided ``on_edge_change`` list of callback functions is
        called whenever a parameter of an edge is changed. Typically, this is used to
        update the transition tensor of the edge or the transition matrix of the
        :py:class:`lymph.models`.

        When a :py:class:`~LymphNodeLevel` is trinary, it is connected to itself via
        a growth edge.
        """
        self._edges: dict[str, Edge] = {}

        for (_, start_name), end_names in graph.items():
            start = self.nodes[start_name]
            if isinstance(start, LymphNodeLevel) and start.is_trinary:
                growth_edge = Edge(parent=start, child=start, callbacks=on_edge_change)
                self._edges[growth_edge.name] = growth_edge

            for end_name in end_names:
                end = self.nodes[end_name]
                new_edge = Edge(parent=start, child=end, callbacks=on_edge_change)
                self._edges[new_edge.name] = new_edge


    @property
    def edges(self) -> dict[str, Edge]:
        """Iterable of all edges in the graph."""
        return self._edges

    @property
    def tumor_edges(self) -> dict[str, Edge]:
        """List of all tumor :py:class:`~Edge` instances in the graph.

        This contains all edges who's parents are instances of :py:class:`~Tumor` and
        who's children are instances of :py:class:`~LymphNodeLevel`.
        """
        return {n: e for n, e in self.edges.items() if e.is_tumor_spread}

    @property
    def lnl_edges(self) -> dict[str, Edge]:
        """List of all LNL :py:class:`~Edge` instances in the graph.

        This contains all edges who's parents and children are instances of
        :py:class:`~LymphNodeLevel` and that are not growth edges.
        """
        return {n: e for n, e in self.edges.items() if not (e.is_tumor_spread or e.is_growth)}

    @property
    def growth_edges(self) -> dict[str, Edge]:
        """List of all growth :py:class:`~Edge` instances in the graph.

        Growth edges are only present in trinary models and are arcs where the parent
        and child are the same :py:class:`~LymphNodeLevel` instance. They facilitate
        the change from a micsoscopically positive to a macroscopically positive LNL.
        """
        return {n: e for n, e in self.edges.items() if e.is_growth}


    def parameter_hash(self) -> int:
        """Compute a hash of the graph.

        Note:
            This is used to check if the graph has changed and the transition matrix
            needs to be recomputed. It should not be used as a replacement for the
            ``__hash__`` method, for two reasons:

            1. It may change over the lifetime of the object, whereas ``__hash__``
                should be constant.
            2. It only takes into account the ``transition_tensor`` of the edges,
                nothing else.

        Example:

        >>> graph_dict = {
        ...    ('tumor', 'T'): ['II', 'III'],
        ...    ('lnl', 'II'): ['III'],
        ...    ('lnl', 'III'): [],
        ... }
        >>> one_graph = Representation(graph_dict)
        >>> another_graph = Representation(graph_dict)
        >>> rng = np.random.default_rng(42)
        >>> for one_edge, another_edge in zip(
        ...     one_graph.edges.values(), another_graph.edges.values()
        ... ):
        ...     params_dict = one_edge.get_params(as_dict=True)
        ...     params_to_set = {k: rng.uniform() for k in params_dict}
        ...     one_edge.set_params(**params_to_set)
        ...     another_edge.set_params(**params_to_set)
        >>> one_graph.parameter_hash() == another_graph.parameter_hash()
        True
        """
        tensor_bytes = b""
        for edge in self.edges.values():
            tensor_bytes += edge.transition_tensor.tobytes()

        return hash(tensor_bytes)


    def to_dict(self) -> dict[tuple[str, str], set[str]]:
        """Returns graph representing this instance's nodes and egdes as dictionary.

        Example:

        >>> graph_dict = {
        ...    ('tumor', 'T'): ['II', 'III'],
        ...    ('lnl', 'II'): ['III'],
        ...    ('lnl', 'III'): [],
        ... }
        >>> graph = Representation(graph_dict)
        >>> graph.to_dict() == graph_dict
        True
        """
        res = {}
        for node in self.nodes.values():
            node_type = "tumor" if isinstance(node, Tumor) else "lnl"
            res[(node_type, node.name)] = [o.child.name for o in node.out]
        return res


    def get_mermaid(self) -> str:
        """Prints the graph in mermaid format.

        Example:

        >>> graph_dict = {
        ...    ('tumor', 'T'): ['II', 'III'],
        ...    ('lnl', 'II'): ['III'],
        ...    ('lnl', 'III'): [],
        ... }
        >>> graph = Representation(graph_dict)
        >>> graph.edges["T_to_II"].spread_prob = 0.1
        >>> graph.edges["T_to_III"].spread_prob = 0.2
        >>> graph.edges["II_to_III"].spread_prob = 0.3
        >>> print(graph.get_mermaid())  # doctest: +NORMALIZE_WHITESPACE
        flowchart TD
            T-->|10%| II
            T-->|20%| III
            II-->|30%| III
        <BLANKLINE>
        """
        mermaid_graph = "flowchart TD\n"

        for node in self.nodes.values():
            for edge in node.out:
                mermaid_graph += f"\t{node.name}-->|{edge.spread_prob:.0%}| {edge.child.name}\n"

        return mermaid_graph


    def get_mermaid_url(self) -> str:
        """Returns the URL to the rendered graph."""
        mermaid_graph = self.get_mermaid()
        graphbytes = mermaid_graph.encode("ascii")
        base64_bytes = base64.b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")
        url="https://mermaid.ink/img/" + base64_string
        return url


    def get_state(self, as_dict: bool = False) -> dict[str, int] | list[int]:
        """Return the states of the system's LNLs.

        If ``as_dict`` is ``True``, the result is a dictionary with the names of the
        LNLs as keys and their states as values. Otherwise, the result is a list of the
        states of the LNLs in the order they appear in the graph.
        """
        result = {}

        for lnl in self.lnls.values():
            result[lnl.name] = lnl.state

        return result if as_dict else list(result.values())


    def set_state(self, *new_states_args, **new_states_kwargs) -> None:
        """Assign a new state to the system's LNLs.

        The state can either be provided with positional arguments or as keyword
        arguments. In case of positional arguments, the order must be the same as the
        order of the LNLs in the graph. If keyword arguments are used, the keys must be
        the names of the LNLs. The order of the keyword arguments does not matter.

        The keyword arguments override the positional arguments.
        """
        for new_lnl_state, lnl in zip(new_states_args, self.lnls.values()):
            lnl.state = new_lnl_state

        for key, value in new_states_kwargs.items():
            lnl = self.nodes[key]
            if lnl is not None and isinstance(lnl, LymphNodeLevel):
                lnl.state = value


    def _gen_state_list(self):
        """Generates the list of (hidden) states."""
        allowed_states_list = []
        for lnl in self.lnls.values():
            allowed_states_list.append(lnl.allowed_states)

        self._state_list = np.array(list(product(*allowed_states_list)))

    @property
    def state_list(self):
        """Return list of all possible hidden states.

        E.g., for three binary LNLs I, II, III, the first state would be where all LNLs
        are in state 0. The second state would be where LNL III is in state 1 and all
        others are in state 0, etc. The third represents the case where LNL II is in
        state 1 and all others are in state 0, etc. Essentially, it looks like binary
        counting:

        >>> graph = Representation(graph_dict={
        ...     ("tumor", "T"): ["I", "II" , "III"],
        ...     ("lnl", "I"): [],
        ...     ("lnl", "II"): ["I", "III"],
        ...     ("lnl", "III"): [],
        ... })
        >>> graph.state_list
        array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 0],
               [0, 1, 1],
               [1, 0, 0],
               [1, 0, 1],
               [1, 1, 0],
               [1, 1, 1]])
        """
        try:
            return self._state_list
        except AttributeError:
            self._gen_state_list()
            return self._state_list
