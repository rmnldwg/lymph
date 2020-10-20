import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import pytest
import emcee
from multiprocessing import Pool
import lymph


@pytest.fixture
def graph():
    graph = {'T': ['II', 'III'],
             'II': ['III'],
             'III': []}
    return graph


@pytest.fixture
def obs_table():
    obs_table = np.array([[[1., 0.],
                           [0., 1.]]])
    return obs_table


@pytest.fixture
def theta():
    """Arbitrarily chosen"""
    theta = np.array([0.2, 0.05, 0.4])
    return theta


@pytest.fixture
def A():
    """Computed by hand"""
    A = np.array([[0.76, 0.04, 0.19, 0.01],
                  [0.00, 0.80, 0.00, 0.20],
                  [0.00, 0.00, 0.57, 0.43],
                  [0.00, 0.00, 0.00, 1.00]])
    return A


@pytest.fixture
def time_prior(max_t=10, p=0.5):
    time_prior = sp.stats.binom.pmf(k=np.linspace(0, max_t, max_t + 1, dtype=int), n=max_t, p=p)
    return time_prior


@pytest.fixture
def data(A, time_prior, samples=10000):
    """Generates a number of samples with a predefined time prior.
    """
    vector = np.array([1., 0., 0., 0.])
    steps = np.zeros(shape=(len(time_prior),len(vector)))

    for i in range(len(time_prior)):
        steps[i,:] = vector
        vector = vector @ A

    states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
    patients = np.zeros(shape=(samples,2), dtype=int)

    for s in range(samples):
        t = np.random.choice(len(time_prior), p=time_prior)
        patients[s,:] = states[np.random.choice(4, p=steps[t,:])]

    columns = pd.MultiIndex.from_arrays([['Info', 'obs', 'obs'],
                                         ['T-stage', 'II', 'III']])
    t_stage = np.asarray(["simple"] * samples).reshape((samples,1))
    data = pd.DataFrame(np.hstack([t_stage, patients]), columns=columns).astype({("obs", "II"): int, ("obs", "III"): int})

    return data


# @pytest.mark.parametrize(
#     "mode",
#     ["BN", "HMM"]
# )
def test_implementation(graph, obs_table, data, theta, A, time_prior, mode="HMM"):
    """Implementing a simple neck system with two lymph node levels.
    """
    # check if transition matrix A gets computed correctly
    neck = lymph.System(graph=graph, obs_table=obs_table)
    neck.set_theta(theta)
    neck.gen_C(data, t_stage=["simple"], observations=["obs"], mode=mode)
    assert np.all(np.isclose(neck.A, A)), "transition matrix erroneous"

    # check if learning works correctly
    nstep, burnin, nwalker, ndim = 1000, 500, 100, 3
    theta0 = np.random.uniform(size=(nwalker, ndim))
    time_dict = {}
    time_dict["simple"] = time_prior
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalker, ndim, neck.likelihood,
                                        args=[["simple"], time_dict, mode],
                                        pool=pool, moves=moves)
        sampler.run_mcmc(theta0, nstep, progress=False)

    samples = sampler.get_chain(flat=True, discard=burnin)
    mean = np.mean(samples, axis=0)
    stddev = np.sqrt(np.var(samples, axis=0))

    for i in range(3):
        print(f"{mean[i]} +- {stddev[i]}")

    assert np.all([np.isclose(mean[i], theta[i], atol=stddev[i]) for i in range(3)]), "learned parameters deviate too much"
