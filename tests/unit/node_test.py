import hypothesis.strategies as st
import numpy as np
import pytest
from custom_strategies import nodes
from hypothesis import HealthCheck, assume, given, settings

from lymph import Edge, Node

settings.register_profile(
    "tests",
    max_examples=10,
    suppress_health_check=HealthCheck.all(),
    deadline=None,
)
settings.load_profile("tests")


@pytest.fixture(scope="session", params=[
    ("detlef", False, "lnl"),
    ("MARIE" , False, "tumor"),
    ("KIM"   , True, "lnl"),
    ("susan" , True, "tumor")
])
def dummy_node(request):
    return Node(*request.param)

@given(
    name=st.text(),
    state=st.one_of(st.integers(), st.booleans()),
    typ=st.one_of(st.text(), st.just("lnl"), st.just("tumor"))
)
def test_constructor(name, state, typ):
    with pytest.raises(TypeError):
        new_node = Node(13, state, typ)

    if int(state) not in [0,1]:
        with pytest.raises(ValueError):
            new_node = Node(name, state, typ)
        return

    if typ not in ["lnl", "tumor"]:
        with pytest.raises(ValueError):
            new_node = Node(name, state, typ)
        return

    new_node = Node(name, state, typ)

    assert new_node.name == name, (
        "Name not correctly assigned"
    )
    assert new_node.typ in ["lnl", "tumor"], (
        "Node has typ that doesn't exist"
    )
    assert new_node.typ == typ, (
        "Typ not correctly assigned"
    )
    assert new_node.state in [0,1], (
        "State of new node must be 0 or 1"
    )
    if new_node.typ == "tumor":
        assert new_node.state == 1, (
            "Tumors must always be in state 1"
        )
    else:
        assert new_node.state == int(state), (
            "State not correctly assigned"
        )
    assert len(new_node.inc) == 0, (
        "Newly created node has incomming connections"
    )
    assert len(new_node.out) == 0, (
        "Newly created node has outgoing connections"
    )
    assert str(new_node) == new_node.name, (
        "String representation does not match with node's name"
    )


@given(
    tumor_node=nodes(typ="tumor", generate_valid=True),
    lnl_node=nodes(typ="lnl", generate_valid=True),
    new_state=st.one_of(st.integers(), st.booleans())
)
def test_state(tumor_node, lnl_node, new_state):
    tumor_node.state = new_state
    assert tumor_node.state == 1, (
        "Tumor node's state must always be 1"
    )

    if int(new_state) not in [0,1]:
        with pytest.raises(ValueError):
            lnl_node.state = new_state
        return

    lnl_node.state = new_state
    assert lnl_node.state == int(new_state), (
        "New state not correctly assigned"
    )

    with pytest.raises(AttributeError):
        del lnl_node._state
        state = lnl_node.state


@given(
    node=nodes(generate_valid=True),
    new_typ=st.one_of(st.text(), st.just("lnl"), st.just("tumor"))
)
def test_typ(node, new_typ):
    if new_typ not in ["lnl", "tumor"]:
        with pytest.raises(ValueError):
            node.typ = new_typ
        return

    node.typ = new_typ

    assert node.typ == new_typ, (
        "New typ not correctly assigned"
    )

    if new_typ == "tumor":
        assert node.state == 1, (
            "When changing typ to `tumor`, state must change to 1"
        )

    with pytest.raises(AttributeError):
        del node._typ
        typ = node.typ


@given(
    inc=st.integers(min_value=0, max_value=10).flatmap(
        lambda n: st.tuples(
            st.tuples(*([st.booleans()] * n)),
            st.tuples(*([st.floats(0., 1.)] * n))
        )
    )
)
def test_trans_prob(inc):
    inc_states, inc_weights = inc

    cached_func_res = Node.trans_prob(inc_states, inc_weights)
    assert len(cached_func_res) == 2, (
        "Must return two transition probabilities"
    )
    assert np.isclose(np.sum(cached_func_res), 1.), (
        "Probabilities must sum to 1"
    )


@given(
    node=nodes(generate_valid=True),
    parents=st.lists(nodes(generate_valid=True), min_size=1, max_size=20),
    spread_probs=st.lists(st.floats(0., 1.), min_size=1, max_size=40),
    log=st.booleans(),
)
def test_bn_prob(node, parents, spread_probs, log):
    """Check the Bayesian net probability"""
    assume(len(spread_probs) >= len(parents))
    for i,parent in enumerate(parents):
        new_edge = Edge(start=parent, end=node, t=spread_probs[i])

    bn_prob = node.bn_prob(log=log)

    if not log:
        assert 0. <= bn_prob <= 1., (
            "Probability must be between zero and one"
        )
    else:
        assert bn_prob <= 0., (
            "Log-probability must be smaller or equal zero"
        )