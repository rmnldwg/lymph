import pytest
import numpy as np
import pandas as pd
import lymph

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
def loaded_sys(sys, data, spsn_dict):
    sys.load_data(data, t_stage=[1,9], spsn_dict=spsn_dict)
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
    
    
def test_load_data(sys, data, spsn_dict, expected_C_dict, expected_f_dict,
                   t_stage=[1,9]):
    sys.load_data(data, t_stage=t_stage, spsn_dict=spsn_dict, mode="HMM")
    
    for t in t_stage:
        assert np.all(np.equal(sys.C_dict[t], expected_C_dict[t]))
        assert np.all(np.equal(sys.f_dict[t], expected_f_dict[t]))
        

def test_likelihood(loaded_sys, t_stage=[1,9]):
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
    

def test_combined_likelihood(loaded_sys, t_stage=[1,9]):
    theta = np.random.uniform(size=(len(loaded_sys.edges)+1))
    c_llh = loaded_sys.combined_likelihood(theta, t_stage=t_stage, T_max=10)
    assert c_llh < 0.
    
    theta = np.random.uniform(size=(len(loaded_sys.edges)+1)) + 1.
    c_llh = loaded_sys.combined_likelihood(theta, t_stage=t_stage, T_max=10)
    assert np.isinf(c_llh)