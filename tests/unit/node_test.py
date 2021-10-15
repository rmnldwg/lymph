import pytest
import lymph


@pytest.fixture(scope="session", params=[
    ("detlef", False, "lnl"),
    ("MARIE" , False, "tumor"),
    ("KIM"   , True, "lnl"),
    ("susan" , True, "tumor")
])
def dummy_node(request):
    return lymph.Node(*request.param)


def test_constructor(dummy_node):
    assert type(dummy_node.name) == str,        "name must be a string"
    assert dummy_node.typ in ["lnl", "tumor"], ("typ of node must be 'lnl' or "
                                                "'tumor'")
    if dummy_node.typ == "tumor":
        assert dummy_node.state == 1,           "tumor node's state must be 1"


@pytest.mark.parametrize("new_state", [0, 1])
def test_state(dummy_node, new_state):
    dummy_node.state = new_state
    if dummy_node.typ == "tumor":
        assert dummy_node.state == 1,           "tumor node's cannot be changed"
    else:
        assert dummy_node.state == new_state,   "node must set new state."


@pytest.mark.parametrize("new_typ", ["lnl", "tumor", "rubbish"])
def test_typ(dummy_node, new_typ):
    if new_typ in ["lnl", "tumor"]:
        dummy_node.typ = new_typ
        assert new_typ == dummy_node.typ,       "node must set new typ"
        if dummy_node.typ == "tumor":
            assert dummy_node.state == 1,      ("Changing typ to tumor did not "
                                                "must change state to 1")
    else:
        with pytest.raises(ValueError):
            dummy_node.typ = new_typ