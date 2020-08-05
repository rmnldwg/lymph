import numpy as np
import pytest
from lymph import Edge, Node

@pytest.fixture
def node_a():
    return Node(name="a")

@pytest.fixture
def node_b():
    return Node(name="b")

def test_initialization(node_a, node_b):
    dummy_edge = Edge(node_a, node_b, t=[0, 0.2])
    assert isinstance(dummy_edge, Edge)