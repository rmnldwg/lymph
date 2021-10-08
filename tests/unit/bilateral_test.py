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
def loaded_bisys(bisys, bidata, t_stage, spsn_dict):
    bisys.load_data(bidata, t_stage=t_stage, spsn_dict=spsn_dict)
    return bisys
    

def test_initialization(bisys):
    assert hasattr(bisys, 'system')
    assert "ipsi" in bisys.system
    assert "contra" in bisys.system
    
    
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
    assert hasattr(bisys.system["ipsi"], 'A')
    for t in range(10):
        row_sums = np.sum(np.linalg.matrix_power(bisys.system["ipsi"].A, t), 
                          axis=1)
        assert np.all(np.isclose(row_sums, 1.))
    
    assert hasattr(bisys.system["contra"], 'A')
    for t in range(10):
        row_sums = np.sum(np.linalg.matrix_power(bisys.system["contra"].A, t), 
                          axis=1)
        assert np.all(np.isclose(row_sums, 1.))
        
    if base_symmetric and trans_symmetric:
        assert np.all(np.equal(bisys.system["ipsi"].A, 
                               bisys.system["contra"].A))
    else:
        assert ~np.all(np.equal(bisys.system["ipsi"].A, 
                                bisys.system["contra"].A))
        

def test_B_matrices(bisys, spsn_dict):
    bisys.set_modalities(spsn_dict=spsn_dict)
    assert hasattr(bisys.system["ipsi"], 'B')
    assert hasattr(bisys.system["contra"], 'B')
    
    row_sums = np.sum(bisys.system["ipsi"].B, axis=1)
    assert np.all(np.isclose(row_sums, 1.))
    
    assert np.all(np.equal(bisys.system["ipsi"].B, bisys.system["contra"].B))
    
    
def test_load_data(bisys, bidata, t_stage, spsn_dict, 
                   expected_C_dict, expected_f_dict):
    bisys.load_data(bidata, t_stage=t_stage, spsn_dict=spsn_dict)
    
    assert hasattr(bisys.system["ipsi"], 'C_dict')
    assert hasattr(bisys.system["ipsi"], 'f_dict')
    assert hasattr(bisys.system["contra"], 'C_dict')
    assert hasattr(bisys.system["contra"], 'f_dict')
    
    for stage in t_stage:
        assert bisys.system["ipsi"].C_dict[stage].shape == bisys.system["contra"].C_dict[stage].shape
    
    
@pytest.mark.parametrize("base_symmetric, trans_symmetric", 
                         [(True, True), 
                          (True, False), 
                          (False, True), 
                          (False, False)])
def test_likelihood(loaded_bisys, t_stage, 
                    base_symmetric, trans_symmetric):
    loaded_bisys.base_symmetric=base_symmetric
    loaded_bisys.trans_symmetric=trans_symmetric
    
    theta = np.random.uniform(size=loaded_bisys.get_theta().shape)
    llh = loaded_bisys.likelihood(theta, t_stage=t_stage, 
                                  time_prior_dict={1: np.ones(shape=(5)) / 5.,
                                                   9: np.ones(shape=(5)) / 5.})
    assert llh < 0.
    
    theta = np.random.uniform(size=loaded_bisys.get_theta().shape) + 1.
    llh = loaded_bisys.likelihood(theta, t_stage=t_stage, 
                                  time_prior_dict={1: np.ones(shape=(5)) / 5.,
                                                   9: np.ones(shape=(5)) / 5.})
    assert np.isinf(llh)
    

@pytest.mark.parametrize("base_symmetric, trans_symmetric", 
                         [(True, True), 
                          (True, False), 
                          (False, True), 
                          (False, False)])
def test_combined_likelihood(loaded_bisys, t_stage,
                             base_symmetric, trans_symmetric):
    loaded_bisys.base_symmetric=base_symmetric
    loaded_bisys.trans_symmetric=trans_symmetric
    
    theta = np.random.uniform(size=(len(loaded_bisys.get_theta())+1))
    c_llh = loaded_bisys.combined_likelihood(theta, t_stage=t_stage, T_max=10)
    assert c_llh < 0.
    
    theta = np.random.uniform(size=(len(loaded_bisys.get_theta())+1)) + 1.
    c_llh = loaded_bisys.combined_likelihood(theta, t_stage=t_stage, T_max=10)
    assert np.isinf(c_llh)
    
    
@pytest.mark.parametrize("inv_ipsi, inv_contra, diag_ipsi, diag_contra", [
    ([True,  False, None],  [None, None,  None],  [False, None,  None],  [None,  None, None]),
    ([False, False, False], [None, True,  True],  [True,  None,  None],  [False, True, None]),
    ([None,  True,  False], [True, True,  True],  [True,  False, False], [None,  True, None]),
    ([False, False, None],  [None, False, False], [False, False, False], [None,  False, None])
])
def test_risk(loaded_bisys, inv_ipsi, inv_contra, diag_ipsi, diag_contra):
    # select random theta
    theta = np.random.uniform(size=loaded_bisys.get_theta().shape)
    
    # use some time-prior
    time_prior = np.ones(5) / 5.
    
    # put together requested involvement & diagnoses in the correct format
    inv_dict = {"ipsi": inv_ipsi, "contra": inv_contra}
    diag_dict = {"ipsi":   {"test-o-meter": diag_ipsi}, 
                 "contra": {"test-o-meter": diag_contra}}
    risk = loaded_bisys.risk(
        theta=theta, 
        inv_dict=inv_dict, 
        diag_dict=diag_dict,
        time_prior=time_prior,
        mode="HMM"
    )
    assert risk >= 0.
    assert risk <= 1.
    
    # the bi- & unilateral risk prediction must be the same, when we ignore one 
    # side in the bilateral case. This means that we provide only ``None`` for 
    # the involvement array of interest for the ignored side and also tell it 
    # that this side's diagnose is missing.
    inv_dict = {"ipsi": inv_ipsi, "contra": [None, None, None]}
    diag_dict = {"ipsi":   {"test-o-meter": diag_ipsi}, 
                 "contra": {"test-o-meter": [None, None, None]}}
    birisk_ignore_contra = loaded_bisys.risk(
        theta=theta,
        inv_dict=inv_dict,
        diag_dict=diag_dict,
        time_prior=time_prior,
        mode="HMM"
    )
    
    inv_dict = {"ipsi": [None, None, None], "contra": inv_contra}
    diag_dict = {"ipsi":   {"test-o-meter": [None, None, None]},
                 "contra": {"test-o-meter": diag_contra}}
    birisk_ignore_ipsi = loaded_bisys.risk(
        theta=theta,
        inv_dict=inv_dict,
        diag_dict=diag_dict,
        time_prior=time_prior,
        mode="HMM"
    )

    ipsi_risk = loaded_bisys.system["ipsi"].risk(
        inv=inv_ipsi,
        diagnoses={"test-o-meter": diag_ipsi},
        time_prior=time_prior,
        mode="HMM"
    )
    
    contra_risk = loaded_bisys.system["contra"].risk(
        inv=inv_contra,
        diagnoses={"test-o-meter": diag_contra},
        time_prior=time_prior,
        mode="HMM"
    )

    assert np.isclose(birisk_ignore_contra, ipsi_risk)
    assert np.isclose(birisk_ignore_ipsi, contra_risk)
    
    # Finally, let's make sure that the ipsilateral risk increases when we 
    # observe more severe contralateral involvement
    inv_dict = {"ipsi": [True, True, True], "contra": [None, None, None]}
    diag_dict = {"ipsi":   {"test-o-meter": [None, None, None]},
                 "contra": {"test-o-meter": [False, False, False]}}
    low_risk = loaded_bisys.risk(
        theta=theta,
        inv_dict=inv_dict,
        diag_dict=diag_dict,
        time_prior=time_prior,
        mode="HMM"
    )
    
    inv_dict = {"ipsi": [True, True, True], "contra": [None, None, None]}
    diag_dict = {"ipsi":   {"test-o-meter": [None, None, None]},
                 "contra": {"test-o-meter": [True, True, True]}}
    high_risk = loaded_bisys.risk(
        theta=theta,
        inv_dict=inv_dict,
        diag_dict=diag_dict,
        time_prior=time_prior,
        mode="HMM"
    )

    assert low_risk < high_risk