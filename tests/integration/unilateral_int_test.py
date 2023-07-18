"""
Implement reasonable scripts that do what a user would normally do to set up
a lymph model.
"""
import numpy as np
import scipy as sp

import lymph


def test_integration():
    """Simple integration test."""
    graph = {
        ("tumor", "T"): ["a", "b", "c"],
        ("lnl", "a"): ["b", "d"],
        ("lnl", "b"): ["c"],
        ("lnl", "c"): ["d"],
        ("lnl", "d"): [],
    }
    model = lymph.Unilateral(graph)

    exp_num_nodes = len(graph)
    exp_num_edges = np.sum([len(val) for val in graph.values()])
    assert len(model.nodes) == exp_num_nodes
    assert len(model.edges) == exp_num_edges
    assert model.graph == graph
    assert len(model.spread_probs) == exp_num_edges

    model.microscopic_parameter = 0.5
    model.growth_parameter = 0.3
    assert model.microscopic_parameter == 0.5
    assert model.growth_parameter == 0.3

    scanners_clinical = {
        "bad_scanner": [0.67, 0.72],
        "fancy_scanner": [0.89, 0.9],
    }
    scanners_pathological = {
        "good_test": [1, 0.91]
    }

    modalities_combined = {
        'clinical' : scanners_clinical,
        'pathological' : scanners_pathological
    }
    model.modalities = modalities_combined

    for mod, spsn in scanners_clinical.items():
        assert mod in model._spsn_tables
        assert np.all(np.sum(model._spsn_tables[mod], axis=0) == 1.)
        assert model._spsn_tables[mod][0,0] == spsn[0]
        assert model._spsn_tables[mod][1,2] == spsn[1]

    for mod, spsn in scanners_pathological.items():
        assert mod in model._spsn_tables
        assert np.all(np.sum(model._spsn_tables[mod], axis=0) == 1.)
        assert model._spsn_tables[mod][0,0] == spsn[0]
        assert model._spsn_tables[mod][1,2] == spsn[1]

    diagnoses = {'bad_scanner': {'a': 1, 'b': 1, 'c' : 1, 'd': 1}, 'fancy_scanner': {'a': 1, 'b': 1, 'c' : 1, 'd': 1}, 'good_test': {'a': 1, 'b': 1, 'c' : 1, 'd': 1}}
    prob = 1.
    model.state = [1,1,1,1]
    for modality, spsn in model._spsn_tables.items():
        if modality in diagnoses:
            mod_diagnose = diagnoses[modality]
            for lnl in model.lnls:
                try:
                    lnl_diagnose = mod_diagnose[lnl.name]
                except KeyError:
                    continue
                except IndexError as idx_err:
                    raise ValueError(
                        "diagnoses were not provided in the correct format"
                    ) from idx_err

                prob *= lnl.obs_prob(lnl_diagnose, spsn)
    assert prob == (1-0.67)**4*(1-0.89)**4*(0.91)**4

    max_t = 10
    time_support = np.arange(max_t + 1)
    time_marg = lymph_trinary.MarginalizorDict(max_t=max_t)
    time_marg["early"] = sp.stats.binom.pmf(time_support, n=max_t, p=0.3)
    time_marg["late"] = lambda t,p: sp.stats.binom.pmf(t, n=max_t, p=p)
    model.diag_time_dists = time_marg
    model.diag_time_dists.update([0.5])

    for stage in ["early", "late"]:
        assert np.all(model.diag_time_dists[stage].support == time_support)
        assert np.isclose(np.sum(model.diag_time_dists[stage].pmf), 1.)

    given_params = np.random.uniform(
        size=len(model.spread_probs) + model.diag_time_dists.num_parametric +2
    )
    model.check_and_assign(given_params)

    exp_dist = {}
    exp_dist["early"] = sp.stats.binom.pmf(time_support, n=max_t, p=0.3)
    exp_dist["late"] = sp.stats.binom.pmf(time_support, n=max_t, p=given_params[-1])
    for stage in ["early", "late"]:
        assert np.all(np.isclose(model.diag_time_dists[stage].pmf, exp_dist[stage]))
    exp_spread_probs = given_params[2:-1]
    assert np.all(model.spread_probs == exp_spread_probs)

    synth_data = model.generate_dataset(
        num_patients=100,
        stage_dist={"early": 0.3, "late": 0.7}
    )
    is_early = synth_data["info", "t_stage"] == "early"
    is_late = synth_data["info", "t_stage"] == "late"
    assert len(synth_data) == 100
    assert np.isclose(np.sum(is_early), 30, atol=50)
    assert np.isclose(np.sum(is_late), 70, atol=50)

    model.patient_data = synth_data
    assert np.all(model.patient_data == synth_data)
    exp_shape_early = (len(model.state_list), np.sum(is_early))
    exp_shape_late = (len(model.state_list), np.sum(is_late))
    assert model.diagnose_matrices["early"].shape == exp_shape_early
    assert model.diagnose_matrices["late"].shape == exp_shape_late

    log_llh = model.likelihood(log=True)
    llh = model.likelihood(log=False)
    assert np.isclose(log_llh, np.log(llh))

    involvement = {
        "a": 0,
        "b": 2,
        "c": 2,
        "d": None,
    }
    involvement1 = {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": None,
    }
    involvement2 = {
        "a": 0,
        "b": 2,
        "c": 1,
        "d": None,
    }
    involvement3 = {
        "a": 0,
        "b": 1,
        "c": 1,
        "d": None,
    }
    bad_diagnosis = {
        "bad_scanner": {
            "a": 0,
            "b": 1,
            "c": 1,
            "d": None,
        }
    }
    better_diagnosis = {
        "fancy_scanner": {
            "a": 0,
            "b": 1,
            "c": 1,
            "d": None,
        }
    }
    best_diagnosis = {
        "good_test": {
            "a": 0,
            "b": 1,
            "c": 1,
            "d": None,
        }
    }
    general_risk = model.risk(involvement=involvement)
    individual_bad_risk = model.risk(
        involvement=involvement,
        given_diagnoses=bad_diagnosis
    )
    individual_better_risk = model.risk(
        involvement=involvement,
        given_diagnoses=better_diagnosis
    )
    individual_best_risk1 = model.risk(
        involvement=involvement,
        given_diagnoses=best_diagnosis
    )
    individual_best_risk0 = model.risk(
        involvement=involvement,
        given_diagnoses=best_diagnosis
    )
    individual_best_risk1 = model.risk(
        involvement=involvement1,
        given_diagnoses=best_diagnosis
    )
    individual_best_risk2 = model.risk(
        involvement=involvement2,
        given_diagnoses=best_diagnosis
    )
    individual_best_risk3 = model.risk(
        involvement=involvement3,
        given_diagnoses=best_diagnosis
    )
    assert 0. <= general_risk <= 1.
    assert individual_bad_risk > general_risk
    assert individual_better_risk > individual_bad_risk
    assert individual_best_risk0+individual_best_risk1+individual_best_risk2+individual_best_risk3 > individual_better_risk
    # we add over all microscopic involvements that are equal to the [0,2,2] state in the macroscopic state for the pathological analysis
