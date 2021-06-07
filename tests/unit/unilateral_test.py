import pytest
import numpy as np
import pandas as pd
import lymph


@pytest.fixture
def t_stage():
    return [1,9]

@pytest.fixture
def spsn_dict():
    return {'test-o-meter': [0.99, 0.88]}

@pytest.fixture
def expected_C_dict():
    return {1: np.array([[0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [1, 0, 1, 0, 0]]),
            9: np.array([[0, 0, 1],
                         [0, 0, 0],
                         [0, 1, 1],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [1, 0, 0],
                         [1, 0, 0]])}
    
@pytest.fixture
def expected_f_dict():
    return {1: np.array([1, 1, 1, 1, 1]),
            9: np.array([1, 1, 2])}
    
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
def loaded_sys(sys, data, t_stage, spsn_dict):
    sys.load_data(data, t_stage=t_stage, spsn_dict=spsn_dict)
    return sys



def test_A_matrix(sys):
    theta = np.random.uniform(size=(len(sys.edges)))
    sys.set_theta(theta)
    assert hasattr(sys, 'A')
    
    for t in range(10):
        row_sums = np.sum(np.linalg.matrix_power(sys.A, t), axis=1)
        assert np.all(np.isclose(row_sums, 1.))
    
    
def test_B_matrix(sys, spsn_dict):
    sys.set_modalities(spsn_dict=spsn_dict)
    assert hasattr(sys, 'B')
    
    row_sums = np.sum(sys.B, axis=1)
    assert np.all(np.isclose(row_sums, 1.))
    
    
def test_load_data(sys, data, t_stage, spsn_dict, 
                   expected_C_dict, expected_f_dict):
    sys.load_data(data, t_stage=t_stage, spsn_dict=spsn_dict, mode="HMM")
    
    for t in t_stage:
        assert np.all(np.equal(sys.C_dict[t], expected_C_dict[t]))
        assert np.all(np.equal(sys.f_dict[t], expected_f_dict[t]))
        

def test_likelihood(loaded_sys, t_stage):
    theta = np.random.uniform(size=(len(loaded_sys.edges)))
    llh = loaded_sys.likelihood(theta, t_stage=t_stage, 
                                time_prior_dict={1: np.ones(shape=(5)) / 5.,
                                                 9: np.ones(shape=(9)) / 9.})
    assert llh < 0.
    
    theta = np.random.uniform(size=(len(loaded_sys.edges))) + 1.
    llh = loaded_sys.likelihood(theta, t_stage=t_stage, 
                                time_prior_dict={1: np.ones(shape=(5)) / 5.,
                                                 9: np.ones(shape=(9)) / 9.})
    assert np.isinf(llh)
    

def test_combined_likelihood(loaded_sys, t_stage):
    theta = np.random.uniform(size=(len(loaded_sys.edges)+1))
    c_llh = loaded_sys.combined_likelihood(theta, t_stage=t_stage, T_max=10)
    assert c_llh < 0.
    
    theta = np.random.uniform(size=(len(loaded_sys.edges)+1)) + 1.
    c_llh = loaded_sys.combined_likelihood(theta, t_stage=t_stage, T_max=10)
    assert np.isinf(c_llh)
    
    
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
    theta = np.random.uniform(size=loaded_sys.get_theta().shape)
    time_prior = np.ones(shape=(10)) / 10.
    
    # new risk with no involvement specified
    risk = loaded_sys.risk(theta, inv=inv, diagnoses=diagnoses, 
                           time_prior=time_prior, mode=mode)
    if inv is None:
        assert len(risk) == len(loaded_sys.state_list)
        assert np.all(np.greater_equal(risk, 0.))
        assert np.all(np.less_equal(risk, 1.))
        assert np.isclose(np.sum(risk), 1.)
    else:
        assert type(risk) == np.float64
        assert risk >= 0. and risk <= 1.