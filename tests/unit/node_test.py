import pytest
import lymph


@pytest.mark.parametrize(
    "name, state, typ",
    [("detlef", False, "lnl"),
     ("MARIE" , False, "tumor"),
     ("KIM"   , True, "lnl"),
     ("susan" , True, "tumor")]
)
def test_node(name, state, typ):
    new_node = lymph.Node(name, state, typ)
    
    assert new_node.name == name
    
    expected_state = True if typ == "tumor" else state
    assert new_node.state == expected_state
    assert new_node.typ == typ
    
    expected_str = f"0 --> {name} ({typ}) --> 0"
    assert str(new_node) == expected_str