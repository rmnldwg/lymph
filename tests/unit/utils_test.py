import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
import scipy as sp
from hypothesis import assume, example, given, settings

from lymph.unilateral import Unilateral
from lymph.utils import (
    change_base,
    comp_state_dist,
    draw_diagnose_times,
    draw_from_simplex,
    fast_binomial_pmf,
    jsondict_to_tupledict,
    system_from_hdf,
    tupledict_to_jsondict,
)


@pytest.fixture
def unilateral_model():
    graph = {
        ('tumor', 'primary'): ['one', 'two'],
        ('lnl', 'one'):       ['two', 'three'],
        ('lnl', 'two'):       ['three'],
        ('lnl', 'three'):     []
    }
    modalities = {'test-o-meter': [0.99, 0.88]}
    patient_data = pd.read_csv(
        "./tests/unilateral_mockup_data.csv",
        header=[0,1]
    )

    model = Unilateral(graph)
    model.modalities = modalities
    model.patient_data = patient_data
    return model


def gen_text_tuples(n):
    """Strategy generating tuples of text."""
    characters = st.characters(blacklist_characters=',')
    text = st.text(alphabet=characters, min_size=1)
    strategy = st.tuples(*([text] * n))
    return strategy

@given(
    st.dictionaries(
        keys=st.integers(1,20).flatmap(gen_text_tuples),
        values=st.one_of(st.floats(), st.integers(), st.text())
    )
)
def test_json_tupledict_interface(tuple_dict):
    assert tuple_dict == jsondict_to_tupledict(tupledict_to_jsondict(tuple_dict)), (
        "Converting round trip did not work"
    )

    with pytest.raises(ValueError):
        tupledict_to_jsondict(
            {
                ("here's a comma,", "in the key-tuple"): 42,
                ("not", "here", "though"): 1337
            }
        )


def test_hdf_io(unilateral_model, tmp_path):
    graph = unilateral_model.graph
    modalities = unilateral_model.modalities
    patient_data = unilateral_model.patient_data

    unilateral_model.to_hdf(
        filename=tmp_path / "test.h5",
        name="model"
    )

    recovered_model = system_from_hdf(
        filename=tmp_path / "test.h5",
        name="model"
    )

    assert recovered_model.graph == graph, (
        "Model graph was not correctly recovered"
    )
    assert recovered_model.modalities == modalities, (
        "Model's modalities were not correctly recovered"
    )
    assert recovered_model.patient_data.equals(patient_data), (
        "Model's stored data was not correclty recovered"
    )


@given(
    k=st.integers(0, 170),
    n=st.integers(0, 170),
    p=st.floats(0., 1.)
)
def test_fast_binomial_pmf(k, n, p):
    assume(k <= n)

    assert np.isclose(fast_binomial_pmf(k, n, p), sp.stats.binom.pmf(k, n, p)), (
        "Binomial PMF is wrong"
    )


@given(
    number=st.integers(-1),
    base=st.integers(-1, 17),
    length=st.integers(0, 1000),
)
@example(number=-1, base=17, length=0)
def test_change_base(number, base, length):
    char_string = "0123456789ABCDEF"

    if number < 0 or base < 2 or base > 16:
        with pytest.raises(ValueError):
            _ = change_base(number, base, False, length)
        with pytest.raises(ValueError):
            _ = change_base(number, base, True, length)
        return

    num_in_new_base = change_base(number, base, False, length)
    assert np.all([char in char_string[:base] for char in num_in_new_base]), (
        "Converted number contains unexpected characters"
    )
    assert len(num_in_new_base) >= length, (
        "Converted number is too short"
    )
    assert num_in_new_base == change_base(number, base, True, length)[::-1], (
        "Reversed string should be the same as the string reversed"
    )

    num_in_new_base = change_base(number, base, False, -abs(length))
    assert num_in_new_base == change_base(number, base, False, None), (
        "Negative length and length=None should yield same result"
    )

    num_in_new_base = change_base(number, base, True, length=0)
    if base < 10 and number > 1:
        assert len(num_in_new_base) >= len(str(number)), (
            "If base < 10, converted number must be longer"
        )

    num_in_new_base = change_base(number, 10, False, length)
    assert int(num_in_new_base) == number, (
        "Cannot recover number of base 10 via python's casting method"
    )


@given(
    table=st.integers(1, 4).flatmap(
        lambda n: npst.arrays(dtype=bool, shape=(100,2**n))
    )
)
@settings(deadline=2000, max_examples=20)
def test_comp_state_dist(table):
    state_dist, state_labels = comp_state_dist(table)

    assert len(state_dist) == len(state_labels), (
        "There must be as many labels as 'bin' ins the histogram"
    )

    for count,label in zip(state_dist, state_labels):
        state = np.array([bool(int(digit)) for digit in label])
        recount = np.sum(np.all(state == table, axis=1))
        assert count == recount, (
            f"Counts don't match up for state {state} and count {count}"
        )


@given(
    num_patients=st.integers(-1, 1000),
    t_stages=st.one_of(
        st.lists(st.integers(0), min_size=1, max_size=20, unique=True),
        st.lists(st.characters(whitelist_categories=('L', 'N'),
                               blacklist_characters=''),
                 min_size=1, max_size=20, unique=True),
        st.lists(st.text(min_size=1), min_size=1, max_size=20, unique=True),
    ),
    max_t=st.integers(1,100)
)
def test_draw_diagnose_times(
    num_patients, t_stages, max_t
):
    num_t_stages = len(t_stages)
    stage_dist = draw_from_simplex(num_t_stages)[0]

    # Generate random diagnose times for each T-stage
    tmp = np.random.randint(low=0, high=max_t, size=num_t_stages)
    diag_times = {t_stage: tmp[i] for i,t_stage in enumerate(t_stages)}

    # Generate random distribution over diagnose time for each T-stage
    tmp = draw_from_simplex(ndim=max_t+1, nsample=num_t_stages)
    time_dists = {t_stage: tmp[i] for i,t_stage in enumerate(t_stages)}

    if num_patients < 1:
        with pytest.raises(ValueError):
            drawn_t_stages, drawn_diagnose_times = draw_diagnose_times(
                num_patients=num_patients,
                stage_dist=stage_dist,
                diag_times=diag_times,
                time_dists=None
            )
        with pytest.raises(ValueError):
            drawn_t_stages, drawn_diagnose_times = draw_diagnose_times(
                num_patients=num_patients,
                stage_dist=stage_dist,
                diag_times=None,
                time_dists=time_dists
            )
        return

    with pytest.raises(ValueError):
        drawn_t_stages, drawn_diagnose_times = draw_diagnose_times(
            num_patients=num_patients,
            stage_dist=stage_dist + 1.,
            diag_times=diag_times,
            time_dists=None
        )
    with pytest.raises(ValueError):
        drawn_t_stages, drawn_diagnose_times = draw_diagnose_times(
            num_patients=num_patients,
            stage_dist=stage_dist,
            diag_times=None,
            time_dists=None
        )

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


@given(
    ndim=st.integers(-1, 1000),
    nsample=st.integers(-1, 1000)
)
def test_draw_from_simplex(ndim, nsample):
    if ndim < 1 or nsample < 1:
        with pytest.raises(ValueError):
            samples = draw_from_simplex(ndim, nsample)
        return

    samples = draw_from_simplex(ndim, nsample)

    assert samples.shape == (nsample, ndim), (
        "Samples have wrong shape"
    )
    assert np.all(np.equal(np.sum(samples, axis=1), 1.)), (
        "Simplex samples must sum to 1"
    )
