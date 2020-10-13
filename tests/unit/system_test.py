import numpy as np
import pytest
import lymph

@pytest.fixture
def graph():
    return {'I': ['II', 'III'], 'II': ['III'], 'III': []}

@pytest.fixture
def obs_table():
    return np.array([[[1., 0.], [0., 1.]]])

@pytest.fixture
def dummy_system(graph, obs_table):
    dummy_system = lymph.System(graph=graph, obs_table=obs_table)
    return dummy_system
    
def test_list_graph(dummy_system, graph):
    assert dummy_system.list_graph() == graph

@pytest.mark.parametrize(
    "newstate",
    [[0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1]]
)
def test_set_state(dummy_system, newstate):
    dummy_system.set_state(newstate=newstate)
    for i,node in enumerate(dummy_system.nodes):
        assert node.state == newstate[i]

# test setting & getting theta
@pytest.mark.parametrize(
    "theta, mode",
    [(np.random.uniform(size=(9,)), "HMM"),
     (np.random.uniform(size=(9,)), "BN")]
)
def test_theta(dummy_system, theta, mode):
    dummy_system.set_theta(theta, mode=mode)
    print(len(dummy_system.get_theta()))
    assert np.all(np.equal(dummy_system.get_theta(), theta))

    if mode == "HMM":
        assert dummy_system.A.shape == (2**len(dummy_system.nodes), 2**len(dummy_system.nodes))