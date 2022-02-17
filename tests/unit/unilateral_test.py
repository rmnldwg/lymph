
import numpy as np
import pytest
from custom_strategies import graphs, models
from hypothesis import assume, given
from hypothesis.strategies import booleans, floats, integers, lists, one_of

from lymph import Edge, Node, Unilateral


@given(graph=graphs())
def test_constructor(graph):
    """Test constructor of base model."""
    # make sure errors are raised for nodes with same name
    for name in [key[1] for key in graph.keys()]:
        if ("tumor", name) in graph and ("lnl", name) in graph:
            with pytest.raises(ValueError):
                model = Unilateral(graph)
                print("was checked")
            return

    model = Unilateral(graph)

    # test nodes
    assert len(model.nodes) == len(graph), (
        "Not enough Nodes instantiated"
    )
    assert len(model.nodes) == len(model.lnls) + len(model.tumors), (
        "Number of tumors and LNLs does not add up"
    )
    assert np.all([lnl.typ == "lnl" for lnl in model.lnls]), (
        "Not all LNL nodes are of typ LNL"
    )
    assert np.all([tumor.typ == "tumor" for tumor in model.tumors]), (
        "Not all tumor nodes are of typ tumor"
    )

    # check edges
    assert len(model.edges) == np.sum([len(val) for val in graph.values()]), (
        "Wrong number of edges"
    )
    assert len(model.edges) == len(model.base_edges) + len(model.trans_edges), (
        "Number of base and trans edges does not add up"
    )

    for edge in model.edges:
        edges_without_current = [e for e in model.edges if e != edge]
        start_without_current = [e.start for e in edges_without_current]
        if edge.start in start_without_current:
            idx = start_without_current.index(edge.start)
            assert edge.end != edges_without_current[idx].end, (
                "Duplicate edges found"
            )

    for key,val in graph.items():
        typ, name = key
        created_node = model.find_node(name)
        assert len(created_node.out) == len(val), (
            f"Number of outgoing edges for node {name} is wrong"
        )
        assert created_node.typ == typ, (
            "Created node has wrong typ"
        )

        if typ == "tumor":
            assert np.all([o in model.base_edges for o in created_node.out]), (
                "Edges going out from tumor nodes must be base edges"
            )
        else:
            assert np.all([o in model.trans_edges for o in created_node.out]), (
                "Edges going out from LNLs must be trans edges"
            )


@given(graph=graphs(unique=True))
def test_string(graph):
    """Test the string representation of the class."""
    model = Unilateral(graph)
    string = str(model)

    for edge in model.edges:
        assert str(edge) in string, (
            "Edge not in string representation"
        )

    model.spread_probs = np.random.uniform(size=model.spread_probs.shape)
    string = str(model)

    for spread_prob in model.spread_probs:
        assert f"{100 * spread_prob:.1f}%" in string, (
            "Spread prob not in string representation"
        )


@given(graph=graphs(unique=True))
def test_find_node_and_find_edge(graph):
    model = Unilateral(graph)

    for _, name in graph.keys():
        found_node = model.find_node(name)
        assert found_node.name == name, (
            "Wrong node found"
        )
        assert type(found_node) == Node, (
            "Found node is not of type Node"
        )
        assert found_node in model.nodes, (
            "Found node not in model network"
        )

    for tpl, cons in graph.items():
        _, name = tpl
        for con in cons:
            found_edge = model.find_edge(name, con)
            assert found_edge.start.name == name, (
                "Start of found edge is wrong"
            )
            assert found_edge.end.name == con, (
                "End of found edge is wrong"
            )
            assert type(found_edge) == Edge, (
                "Found edge is not of type Edge"
            )
            assert found_edge in model.edges, (
                "Found edge not in mode network"
            )


@given(graph=graphs(unique=True))
def test_graph(graph):
    model = Unilateral(graph)
    recovered_graph = model.graph

    for key, val in graph.items():
        assert key in recovered_graph, (
            "Recovered graph is missing a key"
        )
        assert np.all(np.sort(val) == np.sort(recovered_graph[key])), (
            "Recovered graph has wrong connection list"
        )


@given(model=models(), newstate=lists(integers(0, 1)))
def test_state(model, newstate):
    """Check the state assignment"""
    num_lnls = len(model.lnls)

    if len(newstate) < num_lnls:
        with pytest.raises(ValueError):
            model.state = newstate
        return

    model.state = newstate

    assert np.all([s == 0 or s == 1 for s in model.state]), (
        "State is not in {0,1}"
    )
    assert np.all(model.state == newstate[:num_lnls]), (
        "State has not been set correctly"
    )


@given(
    model=models(),
    base_probs=lists(one_of(floats(0., 1.), floats()), min_size=1)
)
def test_base_probs(model, base_probs):
    """Test correct behaviour of base probs"""
    assume(len(model.base_probs) < len(base_probs))
    base_probs = base_probs[:len(model.base_probs)]

    is_larger_than_0 = np.all(np.greater_equal(base_probs, 0.))
    is_smaller_than_1 = np.all(np.less_equal(base_probs, 1.))
    if is_larger_than_0 and is_smaller_than_1:
        tm = model.transition_matrix
        assert hasattr(model, "_transition_matrix"), (
            "Before assigning new base probs, model has no transition matrix"
        )
        model.base_probs = base_probs
        assert np.all(model.base_probs == base_probs), (
            "Base probs have not been assigned correctly"
        )
        assert not hasattr(model, "_transition_matrix"), (
            "Outdated transition matrix has not been deleted"
        )
    else:
        with pytest.raises(ValueError):
            model.base_probs = base_probs

@given(
    model=models(),
    trans_probs=lists(one_of(floats(0., 1.), floats()), min_size=1)
)
def test_trans_probs(model, trans_probs):
    """Test correct behaviour of base probs"""
    assume(len(model.trans_probs) < len(trans_probs))
    trans_probs = trans_probs[:len(model.trans_probs)]

    is_larger_than_0 = np.all(np.greater_equal(trans_probs, 0.))
    is_smaller_than_1 = np.all(np.less_equal(trans_probs, 1.))
    if is_larger_than_0 and is_smaller_than_1:
        tm = model.transition_matrix
        assert hasattr(model, "_transition_matrix"), (
            "Before assigning new trans probs, model has no transition matrix"
        )
        model.trans_probs = trans_probs
        assert np.all(model.trans_probs == trans_probs), (
            "Base probs have not been assigned correctly"
        )
        assert not hasattr(model, "_transition_matrix"), (
            "Outdated transition matrix has not been deleted"
        )
    else:
        with pytest.raises(ValueError):
            model.trans_probs = trans_probs

@given(
    model=models(),
    spread_probs=lists(one_of(floats(0., 1.), floats()), min_size=1)
)
def test_spread_probs(model, spread_probs):
    """Test correct behaviour of base probs"""
    assume(len(model.spread_probs) < len(spread_probs))
    spread_probs = spread_probs[:len(model.spread_probs)]

    is_larger_than_0 = np.all(np.greater_equal(spread_probs, 0.))
    is_smaller_than_1 = np.all(np.less_equal(spread_probs, 1.))
    if is_larger_than_0 and is_smaller_than_1:
        tm = model.transition_matrix
        assert hasattr(model, "_transition_matrix"), (
            "Before assigning new trans probs, model has no transition matrix"
        )
        model.spread_probs = spread_probs
        assert np.all(model.spread_probs == spread_probs), (
            "Base probs have not been assigned correctly"
        )
        base_and_trans = np.concatenate([model.base_probs, model.trans_probs])
        assert np.all(base_and_trans == spread_probs), (
            "Concatenation of base and trans probs must give spread probs"
        )
        assert not hasattr(model, "_transition_matrix"), (
            "Outdated transition matrix has not been deleted"
        )
    else:
        with pytest.raises(ValueError):
            model.spread_probs = spread_probs


@given(
    model=models(state=lists(integers(0,1), min_size=1)),
    newstate=lists(integers(0,1), min_size=1),
    acquire=booleans()
)
def test_comp_transition_prob(model, newstate, acquire):
    """Make sure the probability of transitioning from the current state of
    the network to any other given future state is correct.
    """
    assume(len(model.state) < len(newstate))
    newstate = newstate[:len(model.state)]

    if not np.all([int(s) in [0,1] for s in newstate]):
        with pytest.raises(ValueError):
            transition_prob = model.comp_transition_prob(newstate, acquire)
        return

    transition_prob = model.comp_transition_prob(newstate, acquire)
    assert transition_prob <= 1. and transition_prob >= 0., (
        "Probability cannot be greater than 1 or smaller than 0"
    )
    if np.any(newstate < model.state):
        assert transition_prob == 0., (
            f"Probability for transitions involving self-healing must be 0"
        )
    if acquire:
        assert np.all(model.state == newstate), (
            "Model did not acquire the new state"
        )
    if len(model.state) < 6:
        prob_sum = 0.
        for ns in model.state_list:
            prob_sum += model.comp_transition_prob(ns)
        assert np.isclose(prob_sum, 1.), (
            "All possible transition probs do not sum up to 1"
        )


@given(
    model=models(),
    diagnoses=one_of()
)
def test_comp_diagnose_prob(model, diagnoses):
    """Test the correct computation of the diagnose probability."""
    assert False