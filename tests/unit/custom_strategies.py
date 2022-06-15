import numpy as np
import pandas as pd
from hypothesis import assume
from hypothesis.extra import numpy as hynp
from hypothesis.extra import pandas as hypd
import hypothesis.strategies as st

from lymph import Node
from lymph.unilateral import Unilateral


@st.composite
def nodes(draw, typ=None, generate_valid=False):
    """Define SearchStrategy for Nodes"""
    valid_names = st.text(
        alphabet=st.characters(whitelist_categories='L'),
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
def graphs(draw, min_size=2, max_size=10, unique=True):
    """Define hypothesis strategy for generating graphs"""
    # strategy for names of nodes (both tumors and LNLs)
    st_node_names = st.text(
        alphabet=st.characters(whitelist_categories=['L']),
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
    for i,name in enumerate(node_names):
        connections = draw(st.lists(
            elements=st.sampled_from(node_names[num_tumors:]),
            max_size=max_size,
            unique=unique,
        ))
        connections = connections.remove(name) if name in connections else connections
        connections = [] if connections is None else connections
        if i < num_tumors:
            graph[("tumor", name)] = connections
        else:
            graph[("lnl", name)] = connections

    return graph


@st.composite
def modalities(draw, valid=True, min_size=1, max_size=3):
    """Create SearchStrategy for (valid) modalities."""
    if valid:
        spsn_strategy = st.lists(st.floats(0.5, 1.0), min_size=2, max_size=2)
    else:
        spsn_strategy = st.lists(st.one_of(st.floats(0.5, 1.), st.floats()), min_size=1)

    key_strategy = st.text(
        alphabet=st.characters(whitelist_categories='L'), min_size=1
    )

    dict_strategy = st.dictionaries(
        keys=key_strategy,
        values=spsn_strategy,
        min_size=min_size if valid else 0,
        max_size=max_size if valid else 100,
    )
    return draw(dict_strategy)


@st.composite
def models(draw, states=None, spread_probs=None, modalities=None):
    """Define search strategy for generating unilateral models"""
    graph = draw(graphs())
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
def st_models_and_probs(draw, models=models(), gen_prob_type="base"):
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


def gen_MultiIndex(model: Unilateral) -> pd.Series:
    """Generate a pandas Series diagnose from a diagnose dictionary."""
    modalities = list(model.modalities.keys())
    lnl_names = [lnl.name for lnl in model.lnls]
    return pd.MultiIndex.from_product([modalities, lnl_names])

@st.composite
def st_model_diagnose_tuples(draw, models=models()):
    """Define strategy for a model and a corresponding diagnose"""
    model = draw(models)
    multiindex = gen_MultiIndex(model)
    series = draw(
        hypd.series(
            elements=st.one_of(st.booleans(), st.none()),
            index=st.just(multiindex)
        )
    )
    return (model, series)


@st.composite
def st_model_patientdata_tuples(draw, models=models(), add_t_stages=False):
    """Define search strategy for a tuple of a model and corresponding patient
    data."""
    model = draw(models)
    multiindex = gen_MultiIndex(model)
    patient_data = draw(
        hypd.data_frames(
            columns=hypd.columns(
                multiindex,
                elements=st.one_of(st.none(), st.booleans())
            )
        )
    )

    if add_t_stages:
        t_stages = draw(st.lists(
            st.text(alphabet=st.characters(whitelist_categories='L'), min_size=1),
            min_size=1, max_size=6
        ))
        available_t_stages = st.sampled_from(t_stages)
        t_stage_column = st.lists(
            elements=available_t_stages,
            min_size=len(patient_data),
            max_size=len(patient_data)
        )
        patient_data[("info", "t_stage")] = draw(t_stage_column)

    return (model, patient_data)


@st.composite
def st_t_stages(draw, min_size=1, max_size=20):
    allowed_chars = st.characters(
        whitelist_categories=('L', 'N'), blacklist_characters=''
    )
    return draw(
        st.one_of(
            st.lists(
                st.integers(0),
                min_size=min_size,
                max_size=max_size,
                unique=True
            ),
            st.lists(
                allowed_chars,
                min_size=min_size,
                max_size=max_size,
                unique=True
            ),
            st.lists(
                st.text(alphabet=allowed_chars, min_size=1),
                min_size=min_size,
                max_size=max_size,
                unique=True
            ),
        )
    )

@st.composite
def st_time_dist(draw, min_size=2, max_size=20):
    """Strategy for distributions over diagnose times"""
    n = draw(st.integers(min_size, max_size))
    unnormalized = draw(
        hynp.arrays(
            dtype=float,
            shape=n,
            elements=st.floats(
                min_value=0., exclude_min=True,
                allow_nan=False, allow_infinity=None
            )
        )
    )
    norm = np.sum(unnormalized)
    assume(0. < norm < np.inf)
    return unnormalized / norm

@st.composite
def st_stage_dist_and_time_dists(draw):
    """This strategy generates a tuple of a distribution over T-stages, for
    each of which a time distribution is drawn."""
    t_stages = draw(st_t_stages())

    stage_dist = draw(hynp.arrays(
        dtype=float, shape=len(t_stages), elements=st.floats(0.,1.)
    ))
    norm = np.sum(stage_dist)
    assume(0. < norm < np.inf)
    stage_dist = stage_dist / norm

    max_t = draw(st.integers(0,10))
    time_dists = {}
    for t in t_stages:
        time_dists[t] = draw(hynp.arrays(
            dtype=float, shape=max_t+1, elements=st.floats(0.,1.)
        ))
        norm = np.sum(time_dists[t])
        assume(0. < norm < np.inf)
        time_dists[t] = time_dists[t] / norm

    return stage_dist, time_dists


@st.composite
def logllh_params(
    draw,
    model_patientdata=st_model_patientdata_tuples(
        models=models(modalities=modalities(max_size=2)),
        add_t_stages=True,
    )
):
    """Search strategy for the parameters of the log-likelihood function"""
    model, patient_data = draw(model_patientdata)
    model.patient_data = patient_data
    assume(hasattr(model, "_diagnose_matrices"))

    n = len(model.spread_probs)
    spread_probs = draw(hynp.arrays(dtype=float, shape=n, elements=st.floats(0., 1.)))

    t_stages = st.sampled_from(list(model.diagnose_matrices.keys()))
    len_time_dist = draw(st.integers(1, 20))

    diag_times = draw(
        st.one_of(
            st.none(),
            st.dictionaries(
                keys=t_stages,
                values=st.integers(0, 20),
                min_size=1
            )
        )
    )
    time_dists = draw(
        st.one_of(
            st.none(),
            st.dictionaries(
                keys=t_stages,
                values=st_time_dist(
                    min_size=len_time_dist, max_size=len_time_dist
                ),
                min_size=1
            )
        )
    )
    return (model, spread_probs, diag_times, time_dists)


@st.composite
def st_risk_params(
    draw,
    models=models(spread_probs=st.floats(0., 1.)),
    modalities=modalities(max_size=1)
):
    """Strategy for generating parameters necessary for testing risk function"""
    model = draw(models)
    num_lnls = len(model.lnls)

    modalities = draw(modalities)
    model.modalities = modalities

    st_true_false_none = st.one_of(st.none(), st.booleans())
    involvement = draw(hynp.arrays(
        dtype=object, shape=num_lnls, elements=st_true_false_none
    ))
    diagnoses = {}
    for mod in modalities:
        diagnoses[mod] = draw(hynp.arrays(
            dtype=object, shape=num_lnls, elements=st_true_false_none
        ))
    return (model, involvement, diagnoses)
