import numpy as np
import pandas as pd
from hypothesis import assume
from hypothesis.extra import numpy as hynp
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
def modalities(draw, valid=True, min_size=1, max_size=3):
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
        min_size=min_size if valid else 0,
        max_size=max_size if valid else 100,
    )
    return draw(dict_strategy)


@composite
def models(draw, states=None, spread_probs=None, modalities=None):
    """Define search strategy for generating unilateral models"""
    graph = draw(graphs(max_size=4))
    model = Unilateral(graph)

    if states is not None:
        num_lnls = len(model.lnls)
        model.state = draw(
            hynp.arrays(dtype=int, shape=num_lnls, elements=states)
        )
    if spread_probs is not None:
        len_spread_probs = len(model.spread_probs)
        model.spread_probs = draw(
            hynp.arrays(dtype=float, shape=len_spread_probs, elements=spread_probs)
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


@composite
def t_stages_st(draw, min_size=1, max_size=20):
    allowed_chars = characters(
        whitelist_categories=('L', 'N'), blacklist_characters=''
    )
    return draw(
        one_of(
            lists(
                integers(0),
                min_size=min_size,
                max_size=max_size,
                unique=True
            ),
            lists(
                allowed_chars,
                min_size=min_size,
                max_size=max_size,
                unique=True
            ),
            lists(
                text(alphabet=allowed_chars, min_size=1),
                min_size=min_size,
                max_size=max_size,
                unique=True
            ),
        )
    )

@composite
def time_dist_st(draw, min_size=1, max_size=20):
    """Strategy for distributions over diagnose times"""
    n = draw(integers(min_size, max_size))
    unnormalized = draw(
        hynp.arrays(
            dtype=float,
            shape=n,
            elements=floats(
                min_value=0., exclude_min=True,
                allow_nan=False, allow_infinity=None
            )
        )
    )
    norm = np.sum(unnormalized)
    assume(0. < norm < np.inf)
    return unnormalized / norm

@composite
def stage_dist_and_time_dists(draw):
    """This strategy generates a tuple of a distribution over T-stages, for
    each of which a time distribution is drawn."""
    t_stages = draw(t_stages_st())

    stage_dist = draw(hynp.arrays(
        dtype=float, shape=len(t_stages), elements=floats(0.,1.)
    ))
    norm = np.sum(stage_dist)
    assume(0. < norm < np.inf)
    stage_dist = stage_dist / norm

    max_t = draw(integers(0,10))
    time_dists = {}
    for t in t_stages:
        time_dists[t] = draw(hynp.arrays(
            dtype=float, shape=max_t+1, elements=floats(0.,1.)
        ))
        norm = np.sum(time_dists[t])
        assume(0. < norm < np.inf)
        time_dists[t] = time_dists[t] / norm

    return stage_dist, time_dists


@composite
def logllh_params(
    draw,
    model_patientdata=model_patientdata_tuples(
        models=models(modalities=modalities(max_size=2)),
        add_t_stages=True,
    )
):
    """Search strategy for the parameters of the log-likelihood function"""
    model, patient_data = draw(model_patientdata)
    model.patient_data = patient_data
    assume(hasattr(model, "_diagnose_matrices"))

    n = len(model.spread_probs)
    spread_probs = draw(hynp.arrays(dtype=float, shape=n, elements=floats(0., 1.)))

    t_stages = sampled_from(list(model.diagnose_matrices.keys()))
    len_time_dist = draw(integers(1, 20))

    diag_times = draw(
        one_of(
            none(),
            dictionaries(
                keys=t_stages,
                values=integers(0, 20),
                min_size=1
            )
        )
    )
    time_dists = draw(
        one_of(
            none(),
            dictionaries(
                keys=t_stages,
                values=time_dist_st(
                    min_size=len_time_dist, max_size=len_time_dist
                ),
                min_size=1
            )
        )
    )
    return (model, spread_probs, diag_times, time_dists)


@composite
def risk_params(
    draw,
    models=models(spread_probs=floats(0., 1)),
    modalities=modalities(max_size=1)
):
    """Strategy for generating parameters necessary for testing risk function"""
    model = draw(models)
    model.modalities = draw(modalities)
    num_lnls = len(model.lnls)
    involvement = draw(
        one_of(
            none(),
            lists(
                one_of(none(), booleans()),
                min_size=num_lnls, max_size=num_lnls
            )
        )
    )
    diagnoses = draw(
        dictionaries(
            keys=just(list(model.modalities.keys())[0]),
            values=lists(
                one_of(none(), booleans()),
                min_size=num_lnls, max_size=num_lnls
            ),
            min_size=1, max_size=1
        )
    )
    return (model, involvement, diagnoses)
