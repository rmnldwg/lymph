import numpy as np
import pandas as pd
import pytest
import scipy as sp

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
def unidata():
    return pd.read_csv("./tests/unilateral_mockup_data.csv", header=[0,1])

@pytest.fixture
def bidata():
    return pd.read_csv("./tests/bilateral_mockup_data.csv", header=[0,1,2])

@pytest.fixture
def bisys():
    graph = {('tumor', 'primary'): ['one', 'two'],
             ('lnl', 'one'):       ['two', 'three'],
             ('lnl', 'two'):       ['three'],
             ('lnl', 'three'):     []}
    return lymph.Bilateral(graph=graph)

@pytest.fixture
def loaded_bisys(bisys, bidata, t_stages, modality_spsn):
    bisys.modalities = modality_spsn
    bisys.patient_data = bidata
    return bisys

@pytest.fixture
def spread_probs(bisys):
    return np.random.uniform(low=0., high=1., size=bisys.spread_probs.shape)


def test_initialization(bisys):
    assert hasattr(bisys, 'ipsi')
    assert isinstance(bisys.ipsi, lymph.Unilateral)
    assert hasattr(bisys, 'contra')
    assert isinstance(bisys.contra, lymph.Unilateral)


@pytest.mark.parametrize("base_symmetric, trans_symmetric",
                         [(True, True),
                          (True, False),
                          (False, True),
                          (False, False)])
def test_spread_probs_and_A_matrices(bisys, base_symmetric, trans_symmetric):
    # size of spread_probs depends on symmetries
    spread_probs = np.random.uniform(size=((2 - base_symmetric) * 2
                                    + (2 - trans_symmetric) * 3))

    bisys.base_symmetric = base_symmetric
    bisys.trans_symmetric = trans_symmetric
    bisys.spread_probs = spread_probs

    # input should match read-out
    assert np.all(np.equal(spread_probs, bisys.spread_probs))

    # check A matrices
    assert hasattr(bisys.ipsi, 'transition_matrix')
    for t in range(10):
        row_sums = np.sum(
            np.linalg.matrix_power(bisys.ipsi.transition_matrix, t),
            axis=1
        )
        assert np.all(np.isclose(row_sums, 1.))

    assert hasattr(bisys.contra, 'transition_matrix')
    for t in range(10):
        row_sums = np.sum(
            np.linalg.matrix_power(bisys.contra.transition_matrix, t),
            axis=1
        )
        assert np.all(np.isclose(row_sums, 1.))

    if base_symmetric and trans_symmetric:
        assert np.all(np.equal(bisys.ipsi.transition_matrix,
                               bisys.contra.transition_matrix))
    else:
        assert ~np.all(np.equal(bisys.ipsi.transition_matrix,
                                bisys.contra.transition_matrix))


def test_observation_matrix(bisys, modality_spsn):
    bisys.modalities = modality_spsn
    assert hasattr(bisys.ipsi, 'observation_matrix')
    assert hasattr(bisys.contra, 'observation_matrix')

    row_sums = np.sum(bisys.ipsi.observation_matrix, axis=1)
    assert np.all(np.isclose(row_sums, 1.))

    assert np.all(
        np.equal(bisys.ipsi.observation_matrix, bisys.contra.observation_matrix)
    )


def test_load_data(bisys, bidata, t_stages, modality_spsn):
    bisys.modalities = modality_spsn
    bisys.patient_data = bidata
