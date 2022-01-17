import numpy as np
import pytest

from lymph.utils import draw_diagnose_times, draw_from_simplex


@pytest.mark.parametrize("ndim",    [2,  5,   8,   12])
@pytest.mark.parametrize("nsample", [1, 10, 100, 1000])
def test_draw_from_simplex(ndim, nsample):
    simplex_samples = draw_from_simplex(ndim, nsample)
    if nsample == 1:
        assert simplex_samples.shape == (ndim,)
    else:
        assert simplex_samples.shape == (nsample, ndim), (
            f"Sample should have shape {(nsample, ndim)}, "
            f"but has shape {simplex_samples.shape}"
        )


@pytest.mark.parametrize("num_patients", [10, 50, 100, 1000])
@pytest.mark.parametrize("t_stages", [['a', 'b', 'c'], [1, 2, 3, 4]])
@pytest.mark.parametrize("max_t", [3, 7, 12])
def test_draw_diagnose_times(num_patients, t_stages, max_t):
    stage_dist = draw_from_simplex(len(t_stages))

    # test function when provided with a diganose time for each T-stage
    tmp = np.random.randint(low=0, high=max_t, size=len(t_stages))
    diag_times = {t_stage: tmp[i] for i,t_stage in enumerate(t_stages)}

    drawn_t_stages, drawn_diagnose_times = draw_diagnose_times(
        num_patients=num_patients,
        stage_dist=stage_dist,
        diag_times=diag_times,
        time_dists=None
    )

    assert len(drawn_t_stages) == len(drawn_diagnose_times) == num_patients, (
        "Did not draw the right number of samples."
    )
    assert np.all([t_stage in t_stages for t_stage in drawn_t_stages]), (
        "Drawn T-stages are not from list of provided T-stages."
    )
    assert np.all(np.greater_equal(max_t, drawn_diagnose_times)), (
        "Drawn diagnose times exceed latest set time point."
    )

    # test function when provided with a distribution over diagnose times for
    # each T-stage
    tmp = draw_from_simplex(ndim=max_t+1, nsample=len(t_stages))
    time_dists = {t_stage: tmp[i] for i,t_stage in enumerate(t_stages)}

    drawn_t_stages, drawn_diagnose_times = draw_diagnose_times(
        num_patients=num_patients,
        stage_dist=stage_dist,
        diag_times=None,
        time_dists=time_dists
    )

    assert len(drawn_t_stages) == len(drawn_diagnose_times) == num_patients, (
        "Did not draw the right number of samples."
    )
    assert np.all([t_stage in t_stages for t_stage in drawn_t_stages]), (
        "Drawn T-stages are not from list of provided T-stages."
    )
    assert np.all(np.greater_equal(max_t, drawn_diagnose_times)), (
        "Drawn diagnose times exceed latest set time point."
    )