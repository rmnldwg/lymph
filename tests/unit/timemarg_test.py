import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hynp

from lymph import Marginalizor

settings.register_profile(
    "tests",
    max_examples=10,
    suppress_health_check=HealthCheck.all(),
    deadline=None,
)
settings.load_profile("tests")


@given(
    dist=hynp.arrays(
        dtype=float,
        shape=st.integers(1, 50),
        elements=st.floats(0.1, 100.),
    ),
    max_t=st.integers(1, 50),
    param=st.floats(-10., 10.),
)
def test_marginalizor(dist, max_t, param):
    """
    Test the Marginalizor class.
    """
    if len(dist) != max_t + 1:
        with pytest.raises(ValueError):
            _ = Marginalizor(dist=dist, max_t=max_t)

    fixed_marg = Marginalizor(dist=dist)

    assert np.all(fixed_marg.pmf == dist / np.sum(dist)), (
        "Fixed marg was initialized wrongly"
    )
    assert np.all(fixed_marg.support == np.arange(len(dist))), (
        "Fixed marg has wrong shape"
    )
    assert not fixed_marg.is_updateable, "Fixed marg must not be updateable"
    assert fixed_marg.is_frozen, "Fixed marg must be frozen"

    def func(t,p):
        return np.exp(-(t - p)**2)

    param_marg = Marginalizor(func=func, max_t=max_t)

    assert param_marg.is_updateable, "Parametrized marg must be updateable"
    assert not param_marg.is_frozen, "Parametrized marg must not be frozen yet"
    with pytest.raises(ValueError):
        _ = param_marg.pmf

    param_marg.update(param)
    assert param_marg.is_frozen, "Parametrized marg must be frozen now"

    assert np.isclose(np.sum(param_marg.pmf), 1.), (
        "Parametrized marg not normalized"
    )
    assert param_marg

    with pytest.raises(ValueError):
        _ = Marginalizor(max_t=max_t)
