import pytest
import numpy as np
import pandas as pd
import lymph


@pytest.fixture
def spsn_dict():
    return {'test-o-meter': [0.99, 0.88]}

@pytest.fixture
def unidata():
    return pd.read_csv("./tests/unilateral_mockup_data.csv", header=[0,1])

@pytest.fixture
def bidata():
    return pd.read_csv("./tests/bilateral_mockup_data.csv", header=[0,1,2])

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
def bisys():
    graph = {('tumor', 'primary'): ['one', 'two'],
             ('lnl', 'one'):       ['two', 'three'],
             ('lnl', 'two'):       ['three'],
             ('lnl', 'three'):     []}
    return lymph.BilateralSystem(graph=graph)

@pytest.fixture
def loaded_bisys(bisys, bidata, spsn_dict):
    bisys.load_data(bidata, t_stage=[1,9], spsn_dict=spsn_dict)
    return bisys
    

def test_initialization(bisys):
    assert hasattr(bisys, 'iS')
    assert hasattr(bisys, 'cS')
    
    
@pytest.mark.parametrize("base_symmetric, trans_symmetric", 
                         [(True, True), 
                          (True, False), 
                          (False, True), 
                          (False, False)])
def test_theta_and_A_matrices(bisys, base_symmetric, trans_symmetric):
    # size of theta depends on symmetries
    theta = np.random.uniform(size=((2 - base_symmetric) * 2 
                                    + (2 - trans_symmetric) * 3))
    bisys.set_theta(theta, 
                    base_symmetric=base_symmetric, 
                    trans_symmetric=trans_symmetric)
    
    # input should match read-out
    assert np.all(np.equal(theta, bisys.get_theta()))
    
    # check A matrices
    assert hasattr(bisys.iS, 'A')
    for t in range(10):
        row_sums = np.sum(np.linalg.matrix_power(bisys.iS.A, t), axis=1)
        assert np.all(np.isclose(row_sums, 1.))
    
    assert hasattr(bisys.cS, 'A')
    for t in range(10):
        row_sums = np.sum(np.linalg.matrix_power(bisys.cS.A, t), axis=1)
        assert np.all(np.isclose(row_sums, 1.))
        
    if base_symmetric and trans_symmetric:
        assert np.all(np.equal(bisys.iS.A, bisys.cS.A))
    else:
        assert ~np.all(np.equal(bisys.iS.A, bisys.cS.A))
        

def test_B_matrices(bisys, spsn_dict):
    bisys.set_modalities(spsn_dict=spsn_dict)
    assert hasattr(bisys.iS, 'B')
    assert hasattr(bisys.cS, 'B')
    
    row_sums = np.sum(bisys.iS.B, axis=1)
    assert np.all(np.isclose(row_sums, 1.))
    
    assert np.all(np.equal(bisys.iS.B, bisys.cS.B))
    
    
def test_load_data(bisys, bidata, spsn_dict, expected_C_dict, expected_f_dict, 
                   t_stage=[1,9]):
    bisys.load_data(bidata, t_stage=t_stage, spsn_dict=spsn_dict)
    
    assert hasattr(bisys.iS, 'C_dict')
    assert hasattr(bisys.iS, 'f_dict')
    assert hasattr(bisys.cS, 'C_dict')
    assert hasattr(bisys.cS, 'f_dict')
    
    for stage in t_stage:
        assert bisys.iS.C_dict[stage].shape == bisys.cS.C_dict[stage].shape
    
    
@pytest.mark.parametrize("base_symmetric, trans_symmetric", 
                         [(True, True), 
                          (True, False), 
                          (False, True), 
                          (False, False)])
def test_likelihood(loaded_bisys, base_symmetric, trans_symmetric, 
                    t_stage=[1,9]):
    loaded_bisys.base_symmetric=base_symmetric
    loaded_bisys.trans_symmetric=trans_symmetric
    
    theta = np.random.uniform(size=loaded_bisys.get_theta().shape)
    llh = loaded_bisys.likelihood(theta, t_stage=t_stage, 
                                  time_prior_dict={1: np.ones(shape=(5)) / 5.,
                                                   9: np.ones(shape=(9)) / 9.})
    assert llh < 0.
    
    theta = np.random.uniform(size=loaded_bisys.get_theta().shape) + 1.
    llh = loaded_bisys.likelihood(theta, t_stage=t_stage, 
                                  time_prior_dict={1: np.ones(shape=(5)) / 5.,
                                                   9: np.ones(shape=(9)) / 9.})
    assert np.isinf(llh)