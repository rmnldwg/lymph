import pytest
import lymph


@pytest.mark.parametrize(
    "name, state, typ",
    [("dieter", 0, None),
     ("detlef", 0, "lnl"),
     ("MARIE" , 0, "tumor"),
     ("taylor", 1, None),
     ("KIM"   , 1, "lnl"),
     ("susan" , 1, "tumor")]
)
def test_node(name, state, typ):
    new_node = lymph.Node(name, state, typ)
    
    assert new_node.name == name
    
    expected_state = 1 if typ == "tumor" else state
    assert new_node.state == expected_state
    
    if typ is None:
        expected_typ = "tumor" if name.lower()[0] == 't' else "lnl"
    else:
        expected_typ = typ
    assert new_node.typ == expected_typ
    
    expected_str = f"0 --> {name} ({expected_typ}) --> 0"
    assert str(new_node) == expected_str