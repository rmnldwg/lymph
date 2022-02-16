from enum import unique
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from hypothesis.strategies import (
    SearchStrategy,
    booleans,
    builds,
    characters,
    floats,
    integers,
    just,
    lists,
    none,
    one_of,
    slices,
    text,
)

from lymph import Node
from lymph.unilateral import Unilateral


def nodes(
    is_tumor: bool = False,
    is_lnl: bool = False,
    generate_valid: bool = False
) -> SearchStrategy:
    """Define SearchStrategy for Nodes"""
    reasonable_text = text(alphabet=characters(whitelist_categories='L'),
                           min_size=1)

    if generate_valid:
        name = reasonable_text
        state = one_of(booleans(), integers(0, 1), floats(0., 1.))
        typ = one_of(just("tumor"), just("lnl"))
    else:
        name = one_of(reasonable_text, floats())
        state = one_of(booleans(), integers(), floats(), characters())
        typ = reasonable_text

    if is_tumor:
        typ = just("tumor")
    elif is_lnl:
        typ = just("lnl")

    return builds(Node, name=name, state=state, typ=typ)


def gen_graph(
    node_names: List[str],
    are_tumors: List[bool],
    slice_list: List[slice]
) -> Dict[Tuple[str, str], List[str]]:
    """Deterministically generate a graph for the model based on node names and
    whether they are a tumor or lymph node level.
    """
    max_size = np.min([len(node_names), len(are_tumors), len(slice_list)])
    node_names = node_names[:max_size]
    are_tumors = are_tumors[:max_size]
    slice_list = slice_list[:max_size]

    graph = {}
    for name, is_tumor, slc in zip(node_names, are_tumors, slice_list):
        rem_node_names = [n for n in node_names if n != name]
        graph[("tumor" if is_tumor else "lnl", name)] = rem_node_names[slc]

    return graph

def graphs(
    min_size: int = 0,
    max_size: int = 1000,
    unique: bool = False
) -> SearchStrategy:
    """Define hypothesis strategy for generating graphs"""
    return builds(
        gen_graph,
        node_names=lists(
            text(alphabet=characters(whitelist_categories='L'), min_size=1),
            min_size=min_size, max_size=max_size, unique=unique
        ),
        are_tumors=lists(
            booleans(),
            min_size=min_size, max_size=max_size
        ),
        slice_list=lists(
            slices(max_size),
            min_size=min_size, max_size=max_size
        )
    )


def gen_model(
    graph: Dict[Tuple[str, str], Set[str]],
    spread_probs: Optional[List[float]] = None,
    modalities: Optional[Dict[str, List[float]]] = None,
) -> Unilateral:
    """Generate model from various params"""
    model = Unilateral(graph)

    if spread_probs is not None:
        if len(spread_probs) < len(model.spread_probs):
            nrepeat = (len(model.spread_probs) // len(spread_probs)) + 1
            spread_probs = np.tile(spread_probs, nrepeat)
        model.spread_probs = spread_probs

    if modalities is not None:
        model.modalities = modalities

    return model

def models(
    spread_probs = none(),
    modalities = none()
) -> SearchStrategy:
    """Define hypothesis strategy for generating Unilateral models"""
    return builds(
        gen_model,
        graph=graphs(unique=True),
        spread_probs=spread_probs,
        modalities=modalities,
    )