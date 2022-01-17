import pytest

from lymph import Edge, Node


@pytest.fixture(params=[0, 1], ids=["start_healthy", "start_involved"])
def start_node(request):
    return Node("start", request.param, "lnl")

@pytest.fixture(params=[0, 1], ids=["end_healthy", "end_involved"])
def end_node(request):
    return Node("end", request.param, "lnl")

@pytest.mark.parametrize("t", [0.1, 0.4, 0.85])
def test_edge(start_node, end_node, t):
    """Check basic functionality of Edge class."""
    with pytest.raises(TypeError):
        new_edge = Edge(start="start", end=end_node, t=t)

    with pytest.raises(TypeError):
        new_edge = Edge(start=start_node, end="end", t=t)

    new_edge = Edge(start=start_node, end=end_node, t=t)

    assert new_edge.start == start_node
    assert new_edge.end == end_node
    assert new_edge.t == t
    assert str(new_edge) == f"{start_node}-{100*t:.1f}%->{end_node}"

    assert new_edge in start_node.out
    assert new_edge in end_node.inc