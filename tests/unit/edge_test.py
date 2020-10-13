import numpy as np
import pytest
from lymph import Edge, Node

@pytest.fixture
def node_a():
    return Node(name="a")

@pytest.fixture
def node_b():
    return Node(name="b")

@pytest.fixture
def dummy_edge(node_a, node_b):
    dummy_edge = Edge(node_a, node_b, t=[0, 0.2])
    return dummy_edge

def test_initialization(dummy_edge):
    assert isinstance(dummy_edge, Edge)

# I think this is unneccessary
# def test_report(dummy_edge):
#     assert dummy_edge.report() == "start: a\nend: b\nt = [0, 0.2]"