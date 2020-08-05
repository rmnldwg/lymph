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

# maybe I should do this with parent nodes, so it gets a little more interesting?
@pytest.mark.parametrize(
    "state, p", 
    [(0, 0.0), (0, 0.2), (0, 0.5), (0, 0.7), (0, 1.), 
     (1, 0.0), (1, 0.2), (1, 0.5), (1, 0.7), (1, 1.)]
)
def test_transition(dummy_node, state, p):
    dummy_node.state = state
    dummy_node.p = p
    if state == 0:
        assert np.all(np.equal(dummy_node.trans_prob(), np.array([1-p, p])))
    else:
        assert np.all(np.equal(dummy_node.trans_prob(), np.array([0., 1.])))
    
@pytest.mark.parametrize(
    "state, obs",
    [(0, [0]), (0, [1]), (1, [0]), (1, [1])]
)
def test_observation(dummy_node, state, obs):
    dummy_node.state = state
    assert dummy_node.obs_prob(observation=obs) == dummy_node.obs_table[0,obs[0],state]