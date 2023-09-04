"""
Module defining the nodes and edges of the graph representing the lymphatic system.
"""
from __future__ import annotations

import base64
import warnings
from itertools import product

import numpy as np

from lymph.descriptors import params
from lymph.helper import check_unique_names, trigger


class AbstractNode:
    """Abstract base class for nodes in the graph reprsenting the lymphatic system."""
    def __init__(
        self,
        name: str,
        state: int,
        allowed_states: list[int] | None = None,
    ) -> None:
        """Make a new node.

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


class Edge:
    """This class represents an arc in the graph representation of the lymph system."""
    def __init__(
        self,
        parent: Tumor | LymphNodeLevel,
        child: LymphNodeLevel,
        spread_prob: float = 0.,
        micro_mod: float = 1.,
        callbacks: list[callable] | None = None,
    ):
        """Create a new edge between two nodes.

        The `parent` node must be a `Tumor` or a `LymphNodeLevel`, and the `child` node
        must be a `LymphNodeLevel`.

        The `spread_prob` parameter is the probability of a tumor or involved LNL to
        spread to the next LNL. The `micro_mod` parameter is a modifier for the spread
        probability in case of only a microscopic node involvement.
        """
        self.trigger_callbacks = [self.delete_transition_tensor]
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


    def get_transition_tensor(self) -> np.ndarray:
        """Return the transition tensor of the edge."""
        if not hasattr(self, "_transition_tensor"):
            self._transition_tensor = self.comp_transition_tensor()

        return self._transition_tensor


    def delete_transition_tensor(self) -> None:
        """Delete the transition tensor of the edge."""
        if hasattr(self, "_transition_tensor"):
            del self._transition_tensor


    transition_tensor = property(
        fget=get_transition_tensor,
        fdel=delete_transition_tensor,
        doc="""
        This tensor of the shape (s,e,e) contains the transition probabilities for
        the `Node` at this instance's end to transition from any starting state to
        any new state, given any possible state of the `Node` at the start of this
        edge.

        The correct term can be accessed like this:

        .. code-block:: python

            edge.transition_tensor[start_state, end_state, new_state]
        """
    )


class Representation:
    """Class holding the graph structure of the model.

    This class allows accessing the connected nodes (:py:class:`Tumor` and
    :py:class:`LymphNodeLevel`) and edges (:py:class:`Edge`) of the :py:mod:`models`.
    """
    edge_params = params.GetterSetterAccess()

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
        self._tumors: list[Tumor] = []
        self._lnls: list[LymphNodeLevel] = []

        for node_type, node_name in graph:
            if node_type == "tumor":
                self._tumors.append(
                    Tumor(name=node_name, state=tumor_state)
                )
            elif node_type == "lnl":
                self._lnls.append(
                    LymphNodeLevel(name=node_name, allowed_states=allowed_lnl_states)
                )


    @property
    def tumors(self) -> list[Tumor]:
        """List of all :py:class:`~Tumor` nodes in the graph."""
        return self._tumors

    @property
    def lnls(self) -> list[LymphNodeLevel]:
        """List of all :py:class:`~LymphNodeLevel` nodes in the graph."""
        return self._lnls

    @property
    def nodes(self) -> list[Tumor | LymphNodeLevel]:
        """List of both :py:class:`~Tumor` and :py:class:`~LymphNodeLevel` instances."""
        return self._tumors + self._lnls


    @property
    def allowed_states(self) -> list[int]:
        """Return the list of allowed states for each :py:class:`~LymphNodeLevel`."""
        return self._lnls[0].allowed_states

    @property
    def is_binary(self) -> bool:
        """Indicate if the model is binary.

        Returns ``True`` if all :py:class:`~LymphNodeLevel` instances are binary,
        ``False`` otherwise.
        """
        res = {node.is_binary for node in self._lnls}

        if len(res) != 1:
            raise RuntimeError("Not all lnls have the same number of states")

        return res.pop()

    @property
    def is_trinary(self) -> bool:
        """Returns ``True`` if the graph is trinary, ``False`` otherwise.

        Similar to :py:meth:`~Unilateral.is_binary`."""
        res = {node.is_trinary for node in self._lnls}

        if len(res) != 1:
            raise RuntimeError("Not all lnls have the same number of states")

        return res.pop()


    def find_node(self, name: str) -> Tumor | LymphNodeLevel | None:
        """Finds and returns a node with ``name``."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None


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
        self._tumor_edges: list[Edge] = []
        self._lnl_edges: list[Edge] = []
        self._growth_edges: list[Edge] = []

        for (_, start_name), end_names in graph.items():
            start = self.find_node(start_name)
            if isinstance(start, LymphNodeLevel) and start.is_trinary:
                growth_edge = Edge(parent=start, child=start, callbacks=on_edge_change)
                self._growth_edges.append(growth_edge)

            for end_name in end_names:
                end = self.find_node(end_name)
                new_edge = Edge(parent=start, child=end, callbacks=on_edge_change)

                if new_edge.is_tumor_spread:
                    self._tumor_edges.append(new_edge)
                else:
                    self._lnl_edges.append(new_edge)


    @property
    def tumor_edges(self) -> list[Edge]:
        """List of all tumor :py:class:`~Edge` instances in the graph.

        This contains all edges who's parents are instances of :py:class:`~Tumor` and
        who's children are instances of :py:class:`~LymphNodeLevel`.
        """
        return self._tumor_edges

    @property
    def lnl_edges(self) -> list[Edge]:
        """List of all LNL :py:class:`~Edge` instances in the graph.

        This contains all edges who's parents and children are instances of
        :py:class:`~LymphNodeLevel` and that are not growth edges.
        """
        return self._lnl_edges

    @property
    def growth_edges(self) -> list[Edge]:
        """List of all growth :py:class:`~Edge` instances in the graph.

        Growth edges are only present in trinary models and are arcs where the parent
        and child are the same :py:class:`~LymphNodeLevel` instance. They facilitate
        the change from a micsoscopically positive to a macroscopically positive LNL.
        """
        return self._growth_edges

    @property
    def edges(self) -> list[Edge]:
        """List of all :py:class:`~Edge` instances in the graph, regardless of type."""
        return self._tumor_edges + self._lnl_edges + self._growth_edges


    def find_edge(self, name: str) -> Edge | None:
        """Finds and returns an edge with ``name``."""
        for edge in self.edges:
            if edge.name == name:
                return edge
        return None


    def to_dict(self) -> dict[tuple[str, str], set[str]]:
        """Returns graph representing this instance's nodes and egdes as dictionary."""
        res = {}
        for node in self.nodes:
            node_type = "tumor" if isinstance(node, Tumor) else "lnl"
            res[(node_type, node.name)] = {o.child.name for o in node.out}
        return res


    def get_mermaid(self) -> str:
        """Prints the graph in mermaid format.

        Example:

        >>> graph_dict = {
        ...    ("tumor", "T"): ["II", "III"],
        ...    ("lnl", "II"): ["III"],
        ...    ("lnl", "III"): [],
        ... }
        >>> graph = Representation(graph_dict)
        >>> graph.edge_params["spread_T_to_II"].set_param(0.1)
        >>> graph.edge_params["spread_T_to_III"].set_param(0.2)
        >>> graph.edge_params["spread_II_to_III"].set_param(0.3)
        >>> print(graph.get_mermaid())  # doctest: +NORMALIZE_WHITESPACE
        flowchart TD
            T-->|10%| II
            T-->|20%| III
            II-->|30%| III
        <BLANKLINE>
        """
        mermaid_graph = "flowchart TD\n"

        for idx, node in enumerate(self.nodes):
            for edge in self.nodes[idx].out:
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

        for lnl in self._lnls:
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
        for new_lnl_state, lnl in zip(new_states_args, self._lnls):
            lnl.state = new_lnl_state

        for key, value in new_states_kwargs.items():
            lnl = self.find_node(key)
            if lnl is not None and isinstance(lnl, LymphNodeLevel):
                lnl.state = value


    def _gen_state_list(self):
        """Generates the list of (hidden) states."""
        allowed_states_list = []
        for lnl in self.lnls:
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

        >>> model = Unilateral(graph={
        ...     ("tumor", "T"): ["I", "II" , "III"],
        ...     ("lnl", "I"): [],
        ...     ("lnl", "II"): ["I", "III"],
        ...     ("lnl", "III"): [],
        ... })
        >>> model.state_list
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


    edge_params = params.GetterSetterAccess()
    """Dictionary that maps parameter names to their corresponding parameter objects.

    Parameter names are constructed from the names of the tumors and LNLs in the graph
    that represents the lymphatic system. For example, the parameter for the spread
    probability from the tumor ``T`` to the LNL ``I`` is accessed via the key
    ``spread_T_to_I``.

    The parameters can be read out and changed via the ``get`` and ``set`` methods of
    the :py:class:`~lymph.descriptors.params.Param` objects. The ``set`` method also deletes
    the transition matrix, so that it needs to be recomputed when accessing it the
    next time.

    Example:

    .. code-block:: python

        model.edge_params["spread_T_to_I"].set(0.5)
        retrieved = model.edge_params["spread_T_to_I"].get()
    """
