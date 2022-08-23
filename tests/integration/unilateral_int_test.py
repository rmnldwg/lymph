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

    scanners = {
        "bad_scanner": [0.67, 0.72],
        "fancy_scanner": [0.89, 0.92],
    }
    model.modalities = scanners

    for mod, spsn in scanners.items():
        assert mod in model._spsn_tables
        assert np.all(np.sum(model._spsn_tables[mod], axis=0) == 1.)
        assert model._spsn_tables[mod][0,0] == spsn[0]
        assert model._spsn_tables[mod][1,1] == spsn[1]

    max_t = 10
    time_support = np.arange(max_t + 1)
    time_marg = lymph.MarginalizorDict(max_t=max_t)
    time_marg["early"] = sp.stats.binom.pmf(time_support, n=max_t, p=0.3)
    time_marg["late"] = lambda t,p: sp.stats.binom.pmf(t, n=max_t, p=p)
    model.diag_time_dists = time_marg
    model.diag_time_dists.update([0.5])

    for stage in ["early", "late"]:
        assert np.all(model.diag_time_dists[stage].support == time_support)
        assert np.isclose(np.sum(model.diag_time_dists[stage].pmf), 1.)

    given_params = np.random.uniform(
        size=len(model.spread_probs) + model.diag_time_dists.num_parametric
    )
    model.check_and_assign(given_params)

    exp_dist = {}
    exp_dist["early"] = sp.stats.binom.pmf(time_support, n=max_t, p=0.3)
    exp_dist["late"] = sp.stats.binom.pmf(time_support, n=max_t, p=given_params[-1])
    for stage in ["early", "late"]:
        assert np.all(np.isclose(model.diag_time_dists[stage].pmf, exp_dist[stage]))
    exp_spread_probs = given_params[:-1]
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
        "a": False,
        "b": True,
        "c": True,
        "d": None,
    }
    bad_diagnosis = {
        "bad_scanner": {
            "a": False,
            "b": True,
            "c": True,
            "d": None,
        }
    }
    better_diagnosis = {
        "fancy_scanner": {
            "a": False,
            "b": True,
            "c": True,
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
    assert 0. <= general_risk <= 1.
    assert individual_bad_risk > general_risk
    assert individual_better_risk > individual_bad_risk
