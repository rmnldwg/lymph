
import numpy as np
import pytest
from custom_strategies import graphs, models
from hypothesis import assume, given
from hypothesis.strategies import floats, integers, lists, one_of

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
    assume(len(model.base_probs) < len(base_probs))
    base_probs = base_probs[:len(model.base_probs)]

    is_larger_than_0 = np.all(np.greater(base_probs, 0.))
    is_smaller_than_1 = np.all(np.less(base_probs, 1.))
    if is_larger_than_0 and is_smaller_than_1:
        model.base_probs = base_probs
        assert np.all(model.base_probs == base_probs)
    else:
        with pytest.raises(ValueError):
            model.base_probs = base_probs