import numpy as np
import pytest
from custom_strategies import (
    st_graphs,
    st_likelihood_setup,
    st_modalities,
    st_model_diagnose_tuples,
    st_models,
    st_models_and_data,
    st_models_and_probs,
)
from helpers import are_probabilities
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.strategies import (
    booleans,
    characters,
    floats,
    integers,
    lists,
    none,
    one_of,
    text,
)

from lymph import Edge, Node, Unilateral

settings.register_profile(
    "tests",
    max_examples=10,
    suppress_health_check=HealthCheck.all(),
    deadline=None,
)
settings.load_profile("tests")


@given(graph=st_graphs())
def test_constructor(graph):
    """Test constructor of base model."""
    # make sure errors are raised for nodes with same name
    for name in [key[1] for key in graph.keys()]:
        if ("tumor", name) in graph and ("lnl", name) in graph:
            with pytest.raises(ValueError):
                model = Unilateral(graph)
                print("was checked")
            return

    model = Unilateral(graph)

    # test nodes
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

    # check edges
    assert len(model.edges) == np.sum([len(val) for val in graph.values()]), (
        "Wrong number of edges"
    )
    assert len(model.edges) == len(model.base_edges) + len(model.trans_edges), (
        "Number of base and trans edges does not add up"
    )

    for edge in model.edges:
        edges_without_current = [e for e in model.edges if e != edge]
        start_without_current = [e.start for e in edges_without_current]
        if edge.start in start_without_current:
            idx = start_without_current.index(edge.start)
            assert edge.end != edges_without_current[idx].end, (
                "Duplicate edges found"
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


@given(graph=st_graphs(unique=True))
def test_string(graph):
    """Test the string representation of the class."""
    model = Unilateral(graph)
    string = str(model)

    for edge in model.edges:
        assert str(edge) in string, (
            "Edge not in string representation"
        )

    model.spread_probs = np.random.uniform(size=model.spread_probs.shape)
    string = str(model)

    for spread_prob in model.spread_probs:
        assert f"{100 * spread_prob:.1f}%" in string, (
            "Spread prob not in string representation"
        )


@given(graph=st_graphs(unique=True))
def test_find_node_and_find_edge(graph):
    model = Unilateral(graph)

    for _, name in graph.keys():
        found_node = model.find_node(name)
        assert found_node.name == name, (
            "Wrong node found"
        )
        assert type(found_node) == Node, (
            "Found node is not of type Node"
        )
        assert found_node in model.nodes, (
            "Found node not in model network"
        )

    for tpl, cons in graph.items():
        _, name = tpl
        for con in cons:
            found_edge = model.find_edge(name, con)
            assert found_edge.start.name == name, (
                "Start of found edge is wrong"
            )
            assert found_edge.end.name == con, (
                "End of found edge is wrong"
            )
            assert type(found_edge) == Edge, (
                "Found edge is not of type Edge"
            )
            assert found_edge in model.edges, (
                "Found edge not in mode network"
            )


@given(graph=st_graphs(unique=True))
def test_graph(graph):
    model = Unilateral(graph)
    recovered_graph = model.graph

    for key, val in graph.items():
        assert key in recovered_graph, (
            "Recovered graph is missing a key"
        )
        assert np.all(np.sort(val) == np.sort(recovered_graph[key])), (
            "Recovered graph has wrong connection list"
        )

    assert graph == recovered_graph, "Recovered graph is not the same as original one."


@given(model=st_models(), newstate=lists(integers(0, 1)))
def test_state(model, newstate):
    """Check the state assignment"""
    num_lnls = len(model.lnls)

    if len(newstate) < num_lnls:
        with pytest.raises(ValueError):
            model.state = newstate
        return

    model.state = newstate

    assert np.all([s == 0 or s == 1 for s in model.state]), (
        "State is not in {0,1}"
    )
    assert np.all(model.state == newstate[:num_lnls]), (
        "State has not been set correctly"
    )


@given(
    model_and_base_probs=st_models_and_probs(gen_prob_type="base"),
)
def test_base_probs(model_and_base_probs):
    """Test correct behaviour of base probs"""
    model, base_probs = model_and_base_probs

    is_larger_than_0 = np.all(np.greater_equal(base_probs, 0.))
    is_smaller_than_1 = np.all(np.less_equal(base_probs, 1.))
    if is_larger_than_0 and is_smaller_than_1:
        tm = model.transition_matrix
        assert hasattr(model, "_transition_matrix"), (
            "Before assigning new base probs, model has no transition matrix"
        )
        model.base_probs = base_probs
        assert np.all(model.base_probs == base_probs), (
            "Base probs have not been assigned correctly"
        )
        assert not hasattr(model, "_transition_matrix"), (
            "Outdated transition matrix has not been deleted"
        )
    else:
        with pytest.raises(ValueError):
            model.base_probs = base_probs

@given(
    model_and_trans_probs=st_models_and_probs(gen_prob_type="trans"),
)
def test_trans_probs(model_and_trans_probs):
    """Test correct behaviour of trans probs"""
    model, trans_probs = model_and_trans_probs

    is_larger_than_0 = np.all(np.greater_equal(trans_probs, 0.))
    is_smaller_than_1 = np.all(np.less_equal(trans_probs, 1.))
    if is_larger_than_0 and is_smaller_than_1:
        tm = model.transition_matrix
        assert hasattr(model, "_transition_matrix"), (
            "Before assigning new trans probs, model has no transition matrix"
        )
        model.trans_probs = trans_probs
        assert np.all(model.trans_probs == trans_probs), (
            "Base probs have not been assigned correctly"
        )
        assert not hasattr(model, "_transition_matrix"), (
            "Outdated transition matrix has not been deleted"
        )
    else:
        with pytest.raises(ValueError):
            model.trans_probs = trans_probs

@given(
    model_and_spread_probs=st_models_and_probs(gen_prob_type="all"),
)
def test_spread_probs(model_and_spread_probs):
    """Test correct behaviour of trans probs"""
    model, spread_probs = model_and_spread_probs

    is_larger_than_0 = np.all(np.greater_equal(spread_probs, 0.))
    is_smaller_than_1 = np.all(np.less_equal(spread_probs, 1.))
    if is_larger_than_0 and is_smaller_than_1:
        tm = model.transition_matrix
        assert hasattr(model, "_transition_matrix"), (
            "Before assigning new trans probs, model has no transition matrix"
        )
        model.spread_probs = spread_probs
        assert np.all(model.spread_probs == spread_probs), (
            "Base probs have not been assigned correctly"
        )
        base_and_trans = np.concatenate([model.base_probs, model.trans_probs])
        assert np.all(base_and_trans == spread_probs), (
            "Concatenation of base and trans probs must give spread probs"
        )
        assert not hasattr(model, "_transition_matrix"), (
            "Outdated transition matrix has not been deleted"
        )
    else:
        with pytest.raises(ValueError):
            model.spread_probs = spread_probs


@given(
    model=st_models(states=integers(0,1)),
    newstate=lists(integers(0,1), min_size=1),
    acquire=booleans()
)
def test_comp_transition_prob(model, newstate, acquire):
    """Make sure the probability of transitioning from the current state of
    the network to any other given future state is correct.
    """
    assume(len(model.state) <= len(newstate))
    newstate = newstate[:len(model.state)]

    if not np.all([int(s) in [0,1] for s in newstate]):
        with pytest.raises(ValueError):
            transition_prob = model.comp_transition_prob(newstate, acquire)
        return

    transition_prob = model.comp_transition_prob(newstate, acquire)
    assert transition_prob <= 1. and transition_prob >= 0., (
        "Probability cannot be greater than 1 or smaller than 0"
    )
    if np.any(newstate < model.state):
        assert transition_prob == 0., (
            "Probability for transitions involving self-healing must be 0"
        )
    if acquire:
        assert np.all(model.state == newstate), (
            "Model did not acquire the new state"
        )
    if len(model.state) < 8:
        prob_sum = 0.
        for ns in model.state_list:
            prob_sum += model.comp_transition_prob(ns)
        assert np.isclose(prob_sum, 1.), (
            "All possible transition probs do not sum up to 1"
        )


@given(
    st_model_diagnose_tuples(
        models=st_models(modalities=st_modalities())
    )
)
def test_comp_diagnose_prob(model_and_diagnose):
    """Test the correct computation of the diagnose probability."""
    model, pd_diagnose = model_and_diagnose

    dict_diagnose = {}
    invalid_dict_diagnose = {}
    none_dict_diagnose = {}
    for mod in model.modalities.keys():
        dict_diagnose[mod] = pd_diagnose[mod].to_dict()
        invalid_dict_diagnose[mod] = np.append(pd_diagnose[mod].values, True)
        none_dict_diagnose[mod] = {lnl.name: None for lnl in model.lnls}

    pd_diag_prob = model.comp_diagnose_prob(pd_diagnose)
    dict_diag_prob = model.comp_diagnose_prob(dict_diagnose)
    none_diag_prob = model.comp_diagnose_prob(none_dict_diagnose)
    assert pd_diag_prob == dict_diag_prob, (
        "Same diagnose in different formats must have equal probability"
    )
    assert pd_diag_prob <= 1. and pd_diag_prob >= 0., (
        "Probability must be between 0 and 1"
    )
    assert none_diag_prob == 1., (
        "When all diagnoses are unabserverd, probability must be 1"
    )

    with pytest.raises(ValueError):
        model.comp_diagnose_prob(invalid_dict_diagnose)


@given(model=st_models())
def test_state_list(model):
    assert not hasattr(model, "_state_list"), (
        "Model should not have state list after initialization"
    )

    state_list = model.state_list

    assert hasattr(model, "_state_list"), (
        "Model did not generate state list"
    )
    assert len(state_list) == 2**len(model.lnls), (
        "Wrong number of states"
    )
    assert len(np.unique(state_list, axis=0)) == len(state_list), (
        "Cannot have duplicates in state list"
    )

@given(model=st_models(), modalities=st_modalities())
def test_obs_list(model, modalities):
    assert not hasattr(model, "_obs_list"), (
        "Model should not have obs list after initialization"
    )

    model.modalities = modalities
    obs_list = model.obs_list

    assert hasattr(model, "_obs_list"), (
        "Model did not generate obs list"
    )
    assert len(obs_list) == 2**(len(model.lnls) * len(modalities)), (
        "Wrong number of possible observations"
    )


@given(model=st_models())
def test_allowed_transitions(model):
    """Assert that the mask only allows (the computation of) transitions that
    do not involve self-healing."""
    assert not hasattr(model, "_allowed_transitions"), (
        "Model should not have allowed transitions after initialization"
    )

    num_states = 2**len(model.lnls)
    allowed_transitions = model.allowed_transitions

    assert hasattr(model, "_allowed_transitions"), (
        "Model did not create allowed transitions"
    )
    assert len(allowed_transitions) == len(model.state_list), (
        "Every state must have allowed transitions"
    )
    assert allowed_transitions[0] == list(range(0,len(model.state_list))), (
        "Healthy state must be able to transition into any other state"
    )

    for state_idx, next_state_idxs in allowed_transitions.items():
        model.state = model.state_list[state_idx]
        forbidden_state_idxs = set(range(num_states)).difference(set(next_state_idxs))
        for forbidden_state_idx in forbidden_state_idxs:
            forbidden_state = model.state_list[forbidden_state_idx]
            assert model.comp_transition_prob(newstate=forbidden_state) == 0., (
                "Transition probability to all fobidden states must be zero"
            )


@given(model=st_models())
def test_transition_matrix(model):
    """Verify the properties of the tranistion matrix A"""
    assert not hasattr(model, "_transition_matrix"), (
        "Model should not have transition matrix after initializations"
    )

    A = model.A
    del model._transition_matrix
    transition_matrix = model.transition_matrix

    assert np.all(A == transition_matrix), (
        "`A` and transition matrix must be the same"
    )

    num_states = 2**len(model.lnls)
    assert transition_matrix.shape == (num_states, num_states), (
        "Transition matrix has wrong shape"
    )
    assert np.all(np.isclose(np.sum(transition_matrix, axis=1), 1.)), (
        "Transition matrix must be stochastic matrix (rows sum to 1)"
    )


@given(model=st_models(), modalities=st_modalities())
def test_modalities(model, modalities):
    assert not hasattr(model, "_spsn_tables"), (
        "Model shoud not have spsn tables after initialization"
    )

    flattened_spsn = [s for spsn in modalities.values() for s in spsn]
    has_not_len2 = np.any([len(spsn) != 2 for spsn in modalities.values()])
    is_below_lb = np.any(np.greater(0.5, flattened_spsn))
    is_above_ub = np.any(np.less(1., flattened_spsn))
    is_nan = np.any(np.isnan(flattened_spsn))

    if has_not_len2 or is_below_lb or is_above_ub or is_nan:
        with pytest.raises(ValueError):
            model.modalities = modalities
        return

    model.modalities = modalities

    assert hasattr(model, "_spsn_tables"), (
        "Model did not create spsn tables"
    )
    assert modalities == model.modalities, (
        "Modalities were not recovered correctly"
    )

    for mod, spsn in modalities.items():
        assert mod in model._spsn_tables, (
            "Modality not recognized"
        )
        assert spsn[0] == model._spsn_tables[mod][0,0], (
            "Wrong specificity"
        )
        assert spsn[1] == model._spsn_tables[mod][1,1], (
            "Wrong sensitivity"
        )
        assert np.all(np.isclose(np.sum(model._spsn_tables[mod], axis=0), 1.)), (
            "spsn table must sum to one along columns"
        )

    with pytest.raises(TypeError):
        non_str_modalities = {5: [0.76, 0.82]}
        model.modalities = non_str_modalities

    with pytest.raises(ValueError):
        too_many_vals_modalities = {"foo": [0.51, 0.52, 0.53]}
        model.modalities = too_many_vals_modalities

    with pytest.raises(ValueError):
        out_of_bounds_modalities = {"bar": [-3.7, 42.0]}
        model.modalities = out_of_bounds_modalities


@given(model=st_models(), modalities=st_modalities(max_size=2))
def test_observation_matrix(model, modalities):
    """Make sure the observation matrix is correct"""
    assert not hasattr(model, "_observation_matrix"), (
        "Model should not have observation matrix after initialization"
    )

    model.modalities = modalities

    num_lnls = len(model.lnls)
    num_mod = len(model.modalities)
    observation_matrix = model.observation_matrix

    assert hasattr(model, "_observation_matrix"), (
        "Model did not create observation matrix"
    )
    assert observation_matrix.shape == (2**num_lnls, 2**(num_lnls * num_mod)), (
        "Observation matrix has wrong shape"
    )
    assert np.all(np.isclose(np.sum(observation_matrix, axis=1), 1.)), (
        "Observation matrix must be stochastic matrix (rows sum to 1)"
    )

    model.modalities = {"simple": [1., 1.]}
    observation_matrix = model.observation_matrix

    assert np.all(observation_matrix == np.eye(2**num_lnls)), (
        "For sensitivity & specificity of 100%, observation matrix of only one "
        "modality must be the unit matrix"
    )


@given(
    model_and_table=st_models_and_data(
        models=st_models(modalities=st_modalities())
    ),
    t_stage=one_of(
        integers(),
        characters(whitelist_categories='L'),
        text(alphabet=characters(whitelist_categories='L'), min_size=1)
    )
)
def test_diagnose_matrices(model_and_table, t_stage):
    """Test the generation of the diagnose matrix from a dataset of patients"""
    model, table = model_and_table
    num_lnls = len(model.lnls)
    num_pats = len(table)

    assert not hasattr(model, "_diagnose_matrices"), (
        "Model should not have diagnose matrix after initialization"
    )

    with pytest.raises(AttributeError):
        diagnose_matrices = model.diagnose_matrix

    model._gen_diagnose_matrices(table, t_stage)

    assert hasattr(model, "_diagnose_matrices"), (
        "Model did not create diagnose matrices"
    )

    diagnose_matrices = model.diagnose_matrices

    assert t_stage in diagnose_matrices, (
        "Diagnose matrix not associated with given T-stage"
    )
    assert diagnose_matrices[t_stage].shape == (2**num_lnls, num_pats), (
        "Diagnose matrix has wrong shape"
    )
    assert are_probabilities(diagnose_matrices[t_stage]), (
        "Diagnose matrix must be stochastic (rows sum to 1)"
    )


@given(
    model_and_table=st_models_and_data(
        models=st_models(modalities=st_modalities()),
        add_t_stages=True,
    )
)
def test_patient_data(model_and_table):
    """Check the correct handling of the data."""
    model, patient_data, *_ = model_and_table
    t_stages = set(patient_data[("info", "t_stage")].values)

    assert not hasattr(model, "_patient_data"), (
        "Initialized model should not have patient data"
    )
    with pytest.raises(AttributeError):
        _ = model.patient_data

    model.patient_data = patient_data

    assert hasattr(model, "_patient_data"), (
        "Model should have stored patient data by now"
    )
    assert patient_data.equals(model.patient_data), (
        "Recovered patient data is not the same as the provided"
    )

    if len(patient_data) > 0:
        assert hasattr(model, "_diagnose_matrices"), (
            "Model did not create diagnose matrices"
        )
        for stage in t_stages:
            assert stage in model.diagnose_matrices, (
                "Model did not create the right diagnose matrices for the T-stages"
            )

    del model._spsn_tables
    del model._patient_data

    with pytest.raises(ValueError):
        model.patient_data = patient_data


def test_check_and_assign():
    """Test the function that is supposed to ensure that all params are within bounds.
    """
    assert True


@given(
    model=st_models(),
    spread_probs=lists(floats(0., 1.), min_size=1),
    t_first=integers(0, 10),
    t_last=one_of(none(), integers(1, 20)),
)
def test_evolve(model, spread_probs, t_first, t_last):
    """Assert that the model is evolved correclty over the time steps."""
    assume(len(model.spread_probs) <= len(spread_probs))

    model.spread_probs = spread_probs[:len(model.spread_probs)]

    if t_last is None:
        state_probs = model._evolve(t_first, t_last)

        assert len(state_probs) == 2**len(model.lnls), (
            "Returned state probs have wrong shape"
        )
        assert np.isclose(np.sum(state_probs), 1.), (
            "Sum over probabilities for all states must be 1"
        )
    elif t_first > t_last:
        with pytest.raises(ValueError):
            state_probs = model._evolve(t_first, t_last)
    else:
        state_probs = model._evolve(t_first, t_last)

        assert len(state_probs) == t_last - t_first + 1, (
            "Evolve function computed wrong number of time steps"
        )
        assert state_probs.shape[1] == 2**len(model.lnls), (
            "Returned state probs have wrong shape"
        )
        assert np.all(np.isclose(np.sum(state_probs, axis=1), 1.)), (
            "Sum over probabilities for all states must be 1"
        )


@given(likelihood_setup=st_likelihood_setup())
def test_likelihood(likelihood_setup):
    """Test the likelihood function."""
    model, *likelihood_args = likelihood_setup
    (
        data,
        given_params,
        are_params_valid,
        return_log,
    ) = likelihood_args

    llh = model.likelihood(
        data=data,
        given_params=given_params,
        log=return_log,
    )

    if return_log:
        assert llh <= 0., (
            "log-likelihood must be less-equal 0"
        )
        if not are_params_valid:
            assert llh == -np.inf, (
                "invalid parameters must yield -inf as log-likelihood"
            )
    else:
        assert 0. <= llh <= 1. or np.isclose(0., llh) or np.isclose(1., llh), (
            "likelihood must be between 0 and 1 (or at least close)"
        )
        if not are_params_valid:
            assert llh == 0., (
                "invalid parameters must yield 0 as likelihood"
            )

    model.patient_data = data
    llh_no_data = model.likelihood(given_params=given_params, log=return_log)
    assert llh == llh_no_data, (
        "whether data is loaded in- or outside llh must make no difference"
    )
    if are_params_valid:
        model.check_and_assign(given_params)
        llh_no_params = model.likelihood(log=return_log)
        assert llh_no_params == llh_no_data, (
            "whether parameters are loaded in- or outside llh must make no difference"
        )


@given(
    model=st_models(
        modalities=st_modalities(max_size=2),
        spread_probs=floats(0., 1.)
    ),
    diag_times=lists(integers(0,10), min_size=1, max_size=100),
)
def test_draw_patient_diagnoses(model, diag_times):
    """Check if model correctly draws patient diagnoses"""
    patient_diagnoses = model._draw_patient_diagnoses(diag_times)

    expected_shape = (len(diag_times), len(model.modalities) * len(model.lnls))
    assert patient_diagnoses.shape == expected_shape, (
        f"Patient diagnoses has shape {patient_diagnoses.shape}, but should "
        f"have shape {expected_shape}"
    )
    assert patient_diagnoses.dtype == bool, (
        "Drawn diagnoses should be bools"
    )
