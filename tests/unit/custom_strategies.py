"""
Define custom strategies to generate random samples of various objects
used for testing the functionality of the lymph package.
"""
import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import assume
from hypothesis.extra import numpy as hynp
from hypothesis.extra import pandas as hypd

from lymph import Marginalizor, Node
from lymph.unilateral import Unilateral

ST_CHARACTERS = st.characters(
    whitelist_categories=('L', 'N'),
    blacklist_characters=''
)


def dummy_dist(t,p):
    """A simple unnormalized gauss distribtion as a dummy function."""
    return np.exp(-(t - p)**2)

@st.composite
def st_marginalizor(draw, is_updateable=False, is_frozen=False):
    """Strategy for creating Marginalizor instances."""
    if is_updateable:
        max_t = draw(st.integers(1, 50))
        marg = Marginalizor(func=dummy_dist, max_t=max_t)
        if is_frozen:
            param = draw(st.floats(-10., 10.))
            marg.update(param)
    else:
        dist = hynp.arrays(
            dtype=float,
            shape=draw(st.integers(1,50)),
            elements=st.floats(0., 100.),
        )
        assume(not np.isnan(np.sum(dist)))
        marg = Marginalizor(dist=dist)

    return marg


@st.composite
def nodes(draw, typ=None, generate_valid=False):
    """Define SearchStrategy for Nodes"""
    valid_names = st.text(
        alphabet=ST_CHARACTERS,
        min_size=1
    )
    valid_typs = st.one_of(st.just("tumor"), st.just("lnl"))
    valid_states = st.one_of(st.booleans(), st.integers(0, 1), st.floats(0., 1.))

    if generate_valid:
        name = draw(valid_names)
        typ = draw(valid_typs) if typ is None else typ
        state = draw(valid_states)
    else:
        name = draw(st.one_of(valid_names, st.floats()))
        typ = draw(st.one_of(valid_typs, valid_names)) if typ is None else typ
        state = draw(st.one_of(valid_states, st.characters()))

    return Node(name, state, typ)


@st.composite
def st_graphs(draw, min_size=2, max_size=6, unique=True):
    """Define hypothesis strategy for generating graphs"""
    # strategy for names of nodes (both tumors and LNLs)
    st_node_names = st.text(
        alphabet=ST_CHARACTERS,
        min_size=1
    )

    # draw list of node names from strategy of possible node names
    node_names = draw(st.lists(
        elements=st_node_names,
        min_size=min_size,
        max_size=max_size,
        unique=unique,
    ))

    num_tumors = draw(st.integers(min_value=1, max_value=len(node_names)-1))

    graph = {}
    has_connections = False
    for i,name in enumerate(node_names):
        connections = draw(st.lists(
            elements=st.sampled_from(node_names[num_tumors:]),
            max_size=max_size,
            unique=unique,
        ))
        connections = connections.remove(name) if name in connections else connections
        connections = [] if connections is None else connections
        if len(connections) > 0:
            has_connections = True
        if i < num_tumors:
            graph[("tumor", name)] = connections
        else:
            graph[("lnl", name)] = connections

    assume(has_connections)
    return graph


@st.composite
def st_modalities(draw, min_size=1, max_size=3):
    """Create SearchStrategy for (valid) modalities."""
    modality_names = draw(st.lists(
        elements=ST_CHARACTERS,
        min_size=min_size,
        max_size=max_size,
        unique=True
    ))

    st_spsn = st.floats(0.5, 1.0)
    res = {}
    for mod_name in modality_names:
        res[mod_name] = [draw(st_spsn), draw(st_spsn)]

    return res


@st.composite
def st_models(draw, states=None, spread_probs=None, modalities=None):
    """Define search strategy for generating unilateral models"""
    graph = draw(st_graphs())
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

@st.composite
def st_spread_probs_for_(draw, model, are_values_valid=True, is_shape_valid=True):
    """Strategy for drawing spread probs for a particular model instance."""
    shape = model.spread_probs.shape if is_shape_valid else draw(st.integers(1,100))
    if are_values_valid:
        return draw(hynp.arrays(
            dtype=float,
            shape=shape,
            elements=st.floats(min_value=0., max_value=1.)
        ))
    else:
        res = draw(hynp.arrays(dtype=float, shape=shape))
        res[np.all([0. <= res, res <= 1.], axis=0)] += 1.1
        return res

@st.composite
def st_marg_params_for_(draw, model, are_values_valid=True, is_shape_valid=True):
    """Strategy for drawing parameters of the marginalizors."""
    shape = model.diag_time_dists.num_parametric if is_shape_valid else draw(st.integers(1,100))
    if are_values_valid:
        return draw(hynp.arrays(
            dtype=float,
            shape=shape,
            elements=st.floats(min_value=0., max_value=1.)
        ))
    else:
        res = draw(hynp.arrays(dtype=float, shape=shape))
        res[np.all([0. <= res, res <= 1.], axis=0)] += 1.1
        return res

@st.composite
def st_models_and_probs(draw, models=st_models(), gen_prob_type="base"):
    """
    Strategy for generating a model and a suitable number of spread probs. The
    argument `gen_prob_type` can be 'base', 'trans' or 'all' and depending on the choice
    the strategy will return the appropriate spread probs.
    """
    model = draw(models)

    if gen_prob_type == "base":
        num_probs = len(model.base_probs)
    elif gen_prob_type == "trans":
        num_probs = len(model.trans_probs)
    elif gen_prob_type == "all":
        num_probs = len(model.spread_probs)
    else:
        raise ValueError("Wrong choice of spread prob type.")

    spread_probs = draw(hynp.arrays(
        dtype=float, shape=num_probs, elements=st.floats(0., 1.)
    ))
    return model, spread_probs


def gen_multi_index(model: Unilateral) -> pd.Series:
    """Generate a pandas Series diagnose from a diagnose dictionary."""
    modalities = list(model.modalities.keys())
    lnl_names = [lnl.name for lnl in model.lnls]
    return pd.MultiIndex.from_product([modalities, lnl_names])

@st.composite
def st_model_diagnose_tuples(draw, models=st_models()):
    """Define strategy for a model and a corresponding diagnose"""
    model = draw(models)
    multiindex = gen_multi_index(model)
    series = draw(
        hypd.series(
            elements=st.one_of(st.booleans(), st.none()),
            index=st.just(multiindex)
        )
    )
    return (model, series)


@st.composite
def st_t_stages(draw, min_size=1, max_size=20):
    return draw(
        st.one_of(
            st.lists(
                st.integers(0),
                min_size=min_size,
                max_size=max_size,
                unique=True
            ),
            st.lists(
                ST_CHARACTERS,
                min_size=min_size,
                max_size=max_size,
                unique=True
            ),
            st.lists(
                st.text(alphabet=ST_CHARACTERS, min_size=1),
                min_size=min_size,
                max_size=max_size,
                unique=True
            ),
        )
    )

@st.composite
def st_models_and_data(
    draw,
    models=st_models(modalities=st_modalities()),
    add_t_stages=False
):
    """Define search strategy for a tuple of a model and corresponding patient
    data."""
    model = draw(models)
    multiindex = gen_multi_index(model)
    patient_data = draw(
        hypd.data_frames(
            columns=hypd.columns(
                multiindex,
                elements=st.one_of(st.none(), st.booleans())
            ),
        )
    )
    assume(len(patient_data) > 0)

    if add_t_stages:
        max_t = draw(st.integers(1, 50))
        t_stages = draw(st_t_stages())
        available_t_stages = st.sampled_from(t_stages)
        t_stage_column = st.lists(
            elements=available_t_stages,
            min_size=len(patient_data),
            max_size=len(patient_data)
        )
        patient_data[("info", "t_stage")] = draw(t_stage_column)

        for stage in t_stages:
            model.diag_time_dists[stage] = np.random.uniform(size=max_t+1)

        return model, patient_data, t_stages, max_t

    return model, patient_data

@st.composite
def st_likelihood_setup(draw):
    """Strategy setting up everything needed for testing the likelihood function."""
    model, data, _, max_t = draw(st_models_and_data(add_t_stages=True))
    are_params_valid = draw(st.booleans())
    spread_probs = draw(
        st_spread_probs_for_(model, are_values_valid=are_params_valid)
    )
    marg_params = draw(
        st_marg_params_for_(model, are_values_valid=are_params_valid)
    )
    given_params = np.concatenate([spread_probs, marg_params])
    return_log = draw(st.booleans())

    if model.diag_time_dists.max_t != max_t:
        raise RuntimeError("Max t values disagree")

    return (
        model,
        data,
        given_params,
        are_params_valid,
        return_log
    )
