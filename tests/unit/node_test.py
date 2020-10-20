import numpy as np
import pytest
from lymph import Node


@pytest.fixture
def dummy_node():
    dummy_node = Node(name="test", obs_table=np.array([[[0.9, 0.2],
                                                        [0.1, 0.8]]]))
    return dummy_node


def test_initialization(dummy_node):
    assert isinstance(dummy_node, Node)
