from attr import s
from numpy.lib.npyio import load
import pytest
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import lymph


@pytest.fixture(scope="session")
def t_stages():
    return ["early", "late"]

@pytest.fixture(scope="session")
def max_t():
    return 10

@pytest.fixture(scope="session")
def diag_times(t_stages, max_t):
    res = {}
    for stage in t_stages:
        res[stage] = np.random.randint(low=0, high=max_t)
    return res

@pytest.fixture(scope="session")
def time_dists(t_stages, max_t):
    res = {}
    p = 0.5
    t = np.arange(max_t + 1)
    for stage in t_stages:
        p = np.random.uniform(low=0., high=p)
        res[stage] = sp.stats.binom.pmf(t, max_t, p)
    return res

@pytest.fixture(scope="session", params=[10])
def early_time_dist(request):
    num_time_steps = request.param
    t = np.arange(num_time_steps + 1)
    return sp.stats.binom.pmf(t, num_time_steps, 0.3)

@pytest.fixture(scope="session", params=[10])
def late_time_dist(request):
    num_time_steps = request.param
    t = np.arange(num_time_steps + 1)
    return sp.stats.binom.pmf(t, num_time_steps, 0.7)

@pytest.fixture
def modality_spsn():
    return {'test-o-meter': [0.99, 0.88]}

@pytest.fixture
def expected_C_dict():
    return {"early": np.array([[0, 0, 0, 1, 1],
                               [0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [1, 0, 1, 0, 0]]),
            "late" : np.array([[0, 0, 1],
                               [0, 0, 0],
                               [0, 1, 1],
                               [0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0],
                               [1, 0, 0],
                               [1, 0, 0]])}
    
@pytest.fixture
def expected_f_dict():
    return {"early": np.array([1, 1, 1, 1, 1]),
            "late" : np.array([1, 1, 2])}
    
@pytest.fixture
def data():
    return pd.read_csv("./tests/unilateral_mockup_data.csv", header=[0,1])

@pytest.fixture
def sys():
    graph = {('tumor', 'primary'): ['one', 'two'],
             ('lnl', 'one'):       ['two', 'three'],
             ('lnl', 'two'):       ['three'],
             ('lnl', 'three'):     []}
    return lymph.System(graph=graph)

@pytest.fixture
def loaded_sys(sys, data, t_stages, modality_spsn):
    sys.load_data(data, t_stages=t_stages, modality_spsn=modality_spsn)
    return sys

@pytest.fixture
def spread_probs(sys):
    return np.random.uniform(low=0., high=1., size=sys.spread_probs.shape)


def test_set_and_get_spread_probs(sys):
    new_spread_probs = np.random.uniform(low=0., high=1., size=len(sys.edges))
    sys.spread_probs = new_spread_probs
    assert np.all(np.equal(sys.spread_probs, new_spread_probs))


def test_set_and_get_state(sys):
    newstate = np.random.randint(low=0, high=1, size=len(sys.lnls))
    sys.state = newstate
    assert np.all(np.equal(sys.state, newstate))


def test_A_matrix(sys):
    spread_probs = np.random.uniform(size=(len(sys.edges)))
    sys.spread_probs = spread_probs
    assert hasattr(sys, 'A')
    
    for t in range(10):
        row_sums = np.sum(np.linalg.matrix_power(sys.A, t), axis=1)
        assert np.all(np.isclose(row_sums, 1.))
    
    
def test_B_matrix(sys, modality_spsn):
    sys.modalities = modality_spsn
    assert hasattr(sys, 'B')
    
    row_sums = np.sum(sys.B, axis=1)
    assert np.all(np.isclose(row_sums, 1.))
    
    
def test_load_data(
    sys, data, t_stages, modality_spsn, expected_C_dict, expected_f_dict
):
    sys.load_data(
        data, t_stages=t_stages, modality_spsn=modality_spsn, mode="HMM"
    )
    
    for stage in t_stages:
        assert np.all(np.equal(sys.C_dict[stage], expected_C_dict[stage]))
        assert np.all(np.equal(sys.f_dict[stage], expected_f_dict[stage]))


@pytest.mark.parametrize(
    "marginalize, has_spread_probs_invalid", 
    [(True, True), (False, True), (True, False), (False, False)]
)
def test_log_likelihood(
    loaded_sys, spread_probs, t_stages, diag_times, time_dists, 
    marginalize, has_spread_probs_invalid
):
    """Check the basic likelihood function."""
    if has_spread_probs_invalid:
        spread_probs += 1.
    else:
        with pytest.raises(ValueError):
            assert loaded_sys.log_likelihood(
                spread_probs, t_stages,
                diag_times=None, time_dists=None,
                mode="HMM"
            )
            assert loaded_sys.log_likelihood(
                spread_probs, t_stages,
                diag_times=[], time_dists=None,
                mode="HMM"
            )
            assert loaded_sys.log_likelihood(
                spread_probs, t_stages,
                diag_times=None, time_dists=np.array([]),
                mode="HMM"
            )
    
    if marginalize:
        diag_times = None
    else:
        time_dists = None
        shifted_diag_times = {}
        for stage in t_stages:
            small_shift = np.random.uniform(-0.2, 0.2)
            shifted_diag_times[stage] = diag_times[stage] + small_shift
        
    llh = loaded_sys.log_likelihood(
        spread_probs, t_stages, 
        diag_times=diag_times, time_dists=time_dists,
        mode="HMM"
    )
    assert llh < 0.
    if has_spread_probs_invalid:
        assert np.isinf(llh)

    if not marginalize:
        shifted_llh = loaded_sys.log_likelihood(
            spread_probs, t_stages, 
            diag_times=shifted_diag_times, time_dists=time_dists,
            mode="HMM"
        )
        assert np.isclose(llh, shifted_llh)


def test_marginal_log_likelihood(
    loaded_sys, 
    t_stages,
    early_time_dist,
    late_time_dist
):
    theta = np.random.uniform(size=loaded_sys.spread_probs.shape)
    llh = loaded_sys.marginal_log_likelihood(
        theta, t_stages=t_stages, time_dists={"early": early_time_dist,
                                              "late" : late_time_dist}
    )
    assert llh < 0.
    
    theta = np.random.uniform(size=(len(loaded_sys.edges))) + 1.
    llh = loaded_sys.marginal_log_likelihood(
        theta, t_stages=t_stages, time_dists={"early": early_time_dist,
                                              "late" : late_time_dist}
    )
    assert np.isinf(llh)
    
    theta = np.random.uniform(size=len(loaded_sys.spread_probs) + 3)
    with pytest.raises(ValueError):
        llh = loaded_sys.marginal_log_likelihood(
            theta, t_stages=t_stages, time_dists={"early": early_time_dist,
                                                  "late" : late_time_dist}
        )
    
    

def test_time_log_likelihood(loaded_sys, t_stages):
    spread_probs = np.random.uniform(size=loaded_sys.spread_probs.shape)
    times = np.array([0.7, 3.8])
    theta = np.concatenate([spread_probs, times])
    llh_1 = loaded_sys.time_log_likelihood(
        theta, t_stages=t_stages, max_t=10
    )
    assert llh_1 < 0.
    
    times = np.array([0.8, 3.85])
    theta = np.concatenate([spread_probs, times])
    llh_2 = loaded_sys.time_log_likelihood(
        theta, t_stages=t_stages, max_t=10
    )
    assert np.isclose(llh_1, llh_2)
    
    times = np.array([0.8, 3.4])
    theta = np.concatenate([spread_probs, times])
    llh_3 = loaded_sys.time_log_likelihood(
        theta, t_stages=t_stages, max_t=10
    )
    assert ~np.isclose(llh_1, llh_3)
    
    times = np.array([0.8, 10.6])
    theta = np.concatenate([spread_probs, times])
    llh_4 = loaded_sys.time_log_likelihood(
        theta, t_stages=t_stages, max_t=10
    )
    assert np.isinf(llh_4)
    
    
@pytest.mark.parametrize("inv, diagnoses, mode", [
    (np.array([0,0,0])   , {'test-o-meter': np.array([0,1,0])}   , "HMM"),
    (np.array([None,0,1]), {'test-o-meter': np.array([1,0,0])}   , "HMM"),
    (np.array([0,1,1])   , {'test-o-meter': np.array([0,None,1])}, "HMM"),
    (None                , {'test-o-meter': np.array([0,0,0])}   , "HMM"),
    (np.array([0,0,0])   , {'test-o-meter': np.array([0,1,0])}   , "BN"),
    (np.array([None,0,1]), {'test-o-meter': np.array([1,0,0])}   , "BN"),
    (np.array([0,1,1])   , {'test-o-meter': np.array([0,None,1])}, "BN"),
    (None                , {'test-o-meter': np.array([0,0,0])}   , "BN"),
])
def test_risk(loaded_sys, inv, diagnoses, mode):
    spread_probs = np.random.uniform(size=loaded_sys.spread_probs.shape)
    time_dist = np.ones(shape=(10)) / 10.
    
    # new risk with no involvement specified
    risk = loaded_sys.risk(spread_probs, inv=inv, diagnoses=diagnoses, 
                           time_dist=time_dist, mode=mode)
    if inv is None:
        assert len(risk) == len(loaded_sys.state_list)
        assert np.all(np.greater_equal(risk, 0.))
        assert np.all(np.less_equal(risk, 1.))
        assert np.isclose(np.sum(risk), 1.)
    else:
        assert type(risk) == np.float64
        assert risk >= 0. and risk <= 1.
