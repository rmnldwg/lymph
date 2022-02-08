import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from lymph import Edge, Node
from lymph.node import node_trans_prob


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


@given(
    old_state=st.booleans(),
    new_state=st.one_of(st.integers(), st.booleans())
)
def test_state(old_state, new_state):
    tumor_node = Node(name="atumor", state=old_state, typ="tumor")
    lnl_node = Node(name="alnl", state=old_state, typ="lnl")

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
    old_typ=st.one_of(st.just("lnl"), st.just("tumor")),
    state=st.booleans(),
    new_typ=st.one_of(st.text(), st.just("lnl"), st.just("tumor"))
)
def test_typ(old_typ, state, new_typ):
    node = Node("arbitrary", state, old_typ)

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
    inc_tuple=st.integers(1, 20).flatmap(
        lambda n: st.tuples(
            st.tuples(*([st.booleans()] * n)),
            st.tuples(*([st.floats(0., 1.)] * n))
        )
    ),
    state=st.booleans()
)
def test_trans_prob(inc_tuple, state):
    inc_states, inc_probs = inc_tuple

    target_node = Node("arbitrary", state, "lnl")

    for i, inc in enumerate(zip(inc_states, inc_probs)):
        s, p = inc
        tmp_node = Node(f"inc_{i}", s, "lnl")
        tmp_edge = Edge(tmp_node, target_node, p)

    if int(state) == 1:
        assert target_node.trans_prob() == [0., 1.], (
            "Node in state 1 must remain "
        )
    else:
        method_res = target_node.trans_prob()
        cached_func_res = node_trans_prob(inc_states, inc_probs)
        assert len(method_res) == 2, (
            "Must return two transition probabilities"
        )
        assert np.isclose(np.sum(method_res), 1.), (
            "Probabilities must sum to 1"
        )
        assert np.all(np.equal(method_res, cached_func_res)), (
            "Transition probabilities must be the same for both functions"
        )