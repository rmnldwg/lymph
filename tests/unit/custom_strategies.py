
import pandas as pd
from hypothesis import assume
from hypothesis.extra import pandas as hypd
from hypothesis.strategies import (
    booleans,
    characters,
    composite,
    dictionaries,
    floats,
    integers,
    just,
    lists,
    none,
    one_of,
    sampled_from,
    slices,
    text,
)

from lymph import Node
from lymph.unilateral import Unilateral


@composite
def nodes(draw, typ=None, generate_valid=False):
    """Define SearchStrategy for Nodes"""
    valid_names = text(
        alphabet=characters(whitelist_categories='L'),
        min_size=1
    )
    valid_typs = one_of(just("tumor"), just("lnl"))
    valid_states = one_of(booleans(), integers(0, 1), floats(0., 1.))

    if generate_valid:
        name = draw(valid_names)
        typ = draw(valid_typs) if typ is None else typ
        state = draw(valid_states)
    else:
        name = draw(one_of(valid_names, floats()))
        typ = draw(one_of(valid_typs, valid_names)) if typ is None else typ
        state = draw(one_of(valid_states, characters()))

    return Node(name, state, typ)


@composite
def graphs(draw, min_size=1, max_size=6, unique=True):
    """Define hypothesis strategy for generating graphs"""
    names = text(alphabet=characters(whitelist_categories='L'), min_size=1)

    nodes = draw(
        lists(
            elements=names,
            min_size=min_size,
            max_size=max_size,
            unique=unique
        )
    )

    are_tumors = draw(
        lists(elements=booleans(), min_size=len(nodes), max_size=len(nodes))
    )
    assume(not all(are_tumors))

    slice_list = draw(
        lists(elements=slices(len(nodes)), min_size=len(nodes), max_size=len(nodes))
    )

    graph = {}
    for node, is_tumor, slice in zip(nodes, are_tumors, slice_list):
        key = ("tumor" if is_tumor else "lnl", node)
        rem_nodes = [n for n in nodes if n != node]
        graph[key] = rem_nodes[slice]

    return graph


@composite
def modalities(draw, valid=True):
    """Create SearchStrategy for (valid) modalities."""
    if valid:
        spsn_strategy = lists(floats(0.5, 1.0), min_size=2, max_size=2)
    else:
        spsn_strategy = lists(one_of(floats(0.5, 1.), floats()), min_size=1)

    key_strategy = text(
        alphabet=characters(whitelist_categories='L'), min_size=1
    )

    dict_strategy = dictionaries(
        keys=key_strategy,
        values=spsn_strategy,
        min_size=1 if valid else 0,
        max_size=3 if valid else 100,
    )
    return draw(dict_strategy)


@composite
def models(draw, states=None, spread_probs=None, modalities=None):
    """Define search strategy for generating unilateral models"""
    graph = draw(graphs())
    model = Unilateral(graph)

    if states is not None:
        num_lnls = len(model.lnls)
        model.state = draw(
            lists(elements=states,
                  min_size=num_lnls,
                  max_size=num_lnls)
        )
    if spread_probs is not None:
        len_spread_probs = len(model.spread_probs)
        model.spread_probs = draw(
            lists(elements=spread_probs,
                  min_size=len_spread_probs,
                  max_size=len_spread_probs)
        )
    if modalities is not None:
        model.modalities = draw(modalities)

    return model


def gen_MultiIndex(model: Unilateral) -> pd.Series:
    """Generate a pandas Series diagnose from a diagnose dictionary."""
    modalities = list(model.modalities.keys())
    lnl_names = [lnl.name for lnl in model.lnls]
    return pd.MultiIndex.from_product([modalities, lnl_names])

@composite
def model_diagnose_tuples(draw, models=models()):
    """Define strategy for a model and a corresponding diagnose"""
    model = draw(models)
    multiindex = gen_MultiIndex(model)
    series = draw(
        hypd.series(
            elements=one_of(booleans(), none()),
            index=just(multiindex)
        )
    )
    return (model, series)


@composite
def model_patientdata_tuples(draw, models=models(), add_t_stages=False):
    """Define search strategy for a tuple of a model and corresponding patient
    data."""
    model = draw(models)
    multiindex = gen_MultiIndex(model)
    patient_data = draw(
        hypd.data_frames(
            columns=hypd.columns(
                multiindex,
                elements=one_of(none(), booleans())
            )
        )
    )

    if add_t_stages:
        t_stages = draw(lists(
            text(alphabet=characters(whitelist_categories='L'), min_size=1),
            min_size=1, max_size=6
        ))
        available_t_stages = sampled_from(t_stages)
        t_stage_column = lists(
            elements=available_t_stages,
            min_size=len(patient_data),
            max_size=len(patient_data)
        )
        patient_data[("info", "t_stage")] = draw(t_stage_column)

    return (model, patient_data)