from typing import List

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import characters, lists, text

from lymph import Unilateral


def gen_graph(lnl_names: List[str]):
    graph = {}
    for name in lnl_names:
        is_tumor = np.random.choice([True, False], p=[0.2, 0.8])
        nsubset = np.random.randint(0, len(lnl_names))
        subset = np.random.choice(lnl_names, size=nsubset, replace=False)
        graph[("tumor" if is_tumor else "lnl", name)] = list(subset)
    return graph


@given(
    lnl_names=lists(text(alphabet=characters(whitelist_categories='L'), min_size=1))
)
def test_constructor(lnl_names):
    """Test constructor of base model.
    """
    graph = gen_graph(lnl_names)

    for name in lnl_names:
        if ("tumor", name) in graph and ("lnl", name) in graph:
            with pytest.raises(ValueError):
                model = Unilateral(graph)
                print("was checked")
            return

    model = Unilateral(graph)
    assert len(model.nodes) == len(graph), (
        "Not enough Nodes instantiated"
    )
    assert len(model.nodes) == len(model.lnls) + len(model.tumors), (
        "Number of tumors and LNLs does not add up"
    )
    assert np.all([lnl.typ == "lnl" for lnl in model.lnls]), (
        "Not all LNL nodes are of typ LNL"
    )
    assert np.all([tumor.typ == "tumor" for tumor in model.tumors]), (
        "Not all tumor nodes are of typ tumor"
    )

    assert len(model.edges) == np.sum([len(val) for val in graph.values()]), (
        "Wrong number of edges"
    )
    assert len(model.edges) == len(model.base_edges) + len(model.trans_edges), (
        "Number of base and trans edges does not add up"
    )

    for key,val in graph.items():
        typ, name = key
        created_node = model.find_node(name)
        assert len(created_node.out) == len(val), (
            f"Number of outgoing edges for node {name} is wrong"
        )
        assert created_node.typ == typ, (
            "Created node has wrong typ"
        )

        if typ == "tumor":
            assert np.all([o in model.base_edges for o in created_node.out]), (
                "Edges going out from tumor nodes must be base edges"
            )
        else:
            assert np.all([o in model.trans_edges for o in created_node.out]), (
                "Edges going out from LNLs must be trans edges"
            )