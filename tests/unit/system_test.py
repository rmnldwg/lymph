import numpy as np
import pytest
import lymph


def test_wrong_graph():
    """Check that given an inconsistent graph, the appropriate error is raised.
    """
    wrong_graph = {"A": ["x", "y", "z"], 
                   "B": ["omega"], 
                   "C": []}
    
    with pytest.raises(ValueError) as e_info:
        wrong_sys = lymph.System(graph=wrong_graph)
    
    assert e_info.value.args[0] == ("Every entry in the list of child nodes "
                                     "must exist as a key in the dictionary "
                                     "as well!")
    
    
@pytest.fixture
def graph():
    return {"T": ["a", "b", "c"], 
            "a": ["b", "c"], 
            "b": ["c"], 
            "c": []}
    
    
def test_system_basics(graph):
    """Check some basic functionalities of the System class."""
    sys = lymph.System(graph=graph)
    
    # check if the graph is reproduced correctly
    assert sys.get_graph() == graph
    
    # check if all the names are correct
    for i,key in enumerate(graph):
        assert sys.nodes[i].name == key
        
    # check if list_edges() reports the correct number of edges
    ends = []
    for key in graph:
        ends.extend(graph[key])
    assert len(ends) == len(sys.list_edges())
    

@pytest.mark.parametrize(
    "newstate", 
    [([0, 0, 0]), 
     ([0, 0, 1]), 
     ([0, 1, 0]), 
     ([0, 1, 1]), 
     ([1, 0, 0]), 
     ([1, 0, 1]), 
     ([1, 1, 0]), 
     ([1, 1, 1])]
)
def test_set_state(graph, newstate):
    """Check set_state() method."""
    sys = lymph.System(graph=graph)
    
    sys.set_state(newstate)
    assert sys.tumors[0].state == 1
    for i in range(3):
        assert sys.lnls[i].state == newstate[i]
        
        
@pytest.mark.parametrize(
    "theta", 
    [([0., 0., 0., 0., 0., 0.]), 
     ([1., 1., 1., 1., 1., 1.]), 
     ([.7, .5, .3, .1, .5, .9])]
)
def test_setget_theta(graph, theta):
    """Check if set_theta() and get_theta() do what they are supposed to do."""
    sys = lymph.System(graph=graph)
    sys.set_theta(theta)
    
    assert np.all(sys.get_theta() == theta)
    
    direct_theta = [edge.t for edge in sys.edges]
    assert np.all(direct_theta == theta)