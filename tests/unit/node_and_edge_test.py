import numpy as np
import pytest
from lymph import Node, Edge


@pytest.mark.parametrize(
    "name, expected_type, expected_state",
    [("tumor", "tumor", 1),
     ("tante", "tumor", 1),
     ("TTT", "tumor", 1),
     ("lnl", "lnl", 0),
     ("viovnrvnrel", "lnl", 0)]
)
def test_node_initialization(name, expected_type, expected_state):
    """Check initialization and if nodes are assigned the correct type."""
    node_instance = Node(name)
    assert node_instance.name == name
    assert node_instance.typ == expected_type
    assert node_instance.state == expected_state


@pytest.fixture
def parent_tumor():
    return Node("parent_tumor", typ="tumor")


@pytest.fixture
def parent_lnl():
    return Node("parent_lnl")


@pytest.fixture
def child_lnl():
    return Node("child_lnl")


@pytest.mark.parametrize(
    "b, parent_lnl_state, t, expected_trans_prob",
    [(0. , 0, 0. , [1., 0.]),
     (1. , 0, 0. , [0., 1.]),
     (1. , 0, 1. , [0., 1.]),
     (1. , 1, 1. , [0., 1.]),
     (0.5, 0, 0. , [0.5, 0.5]),
     (0.5, 1, 0.5, [0.25, 0.75])]
)
def test_trans_prob(parent_tumor, b,
                    parent_lnl, parent_lnl_state, t,
                    child_lnl, expected_trans_prob):
    """Test the transition probability inside the Node instance."""
    # no need to append Edge to the list of incoming connections of the child
    # node, since that is automatically done during the initialization of the 
    # connection.
    base_edge = Edge(start=parent_tumor, end=child_lnl, t=b)

    parent_lnl.state = parent_lnl_state
    trans_edge = Edge(start=parent_lnl, end=child_lnl, t=t)

    assert np.all(np.equal(child_lnl.trans_prob(), expected_trans_prob))

    child_lnl.state = 1   # when node is already involved, it should remain so
    assert np.all(np.equal(child_lnl.trans_prob(), [0., 1.]))