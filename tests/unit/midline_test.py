import numpy as np
import pandas as pd
import pytest

import lymph


@pytest.fixture
def data():
    path_dict = {
        "bilateral": "./tests/bilateral_mockup_data.csv",
        "midext": "./tests/midline_ext_mockup_data.csv"
    }
    return {name: pd.read_csv(path, header=[0,1,2]) for name, path in path_dict.items()}

@pytest.fixture(scope="session")
def t_stages():
    return ["early", "late"]

@pytest.fixture
def modality_spsn():
    return {'test-o-meter': [0.99, 0.88]}

@pytest.fixture
def bisys():
    graph = {('tumor', 'primary'): ['one', 'two'],
             ('lnl', 'one'):       ['two', 'three'],
             ('lnl', 'two'):       ['three'],
             ('lnl', 'three'):     []}
    return lymph.Bilateral(graph=graph)

@pytest.fixture
def midbi():
    graph = {('tumor', 'primary'): ['one', 'two'],
             ('lnl', 'one'):       ['two', 'three'],
             ('lnl', 'two'):       ['three'],
             ('lnl', 'three'):     []}
    return lymph.MidlineBilateral(graph=graph)

@pytest.fixture
def loaded_midbi(data, t_stages, modality_spsn):
    graph = {('tumor', 'primary'): ['one', 'two'],
             ('lnl', 'one'):       ['two', 'three'],
             ('lnl', 'two'):       ['three'],
             ('lnl', 'three'):     []}
    midbi = lymph.MidlineBilateral(graph=graph)
    midbi.modalities = modality_spsn
    midbi.patient_data = data["midext"]
    return midbi

@pytest.fixture
def new_spread_probs(midbi):
    return np.random.uniform(size=midbi.spread_probs.shape)


def test_spread_probs(midbi, new_spread_probs):
    midbi.spread_probs = new_spread_probs

    assert np.all(np.equal(new_spread_probs, midbi.spread_probs)), (
        "Spread probabilities haven't been set correctly."
    )
    assert np.all(np.equal(midbi.noext.ipsi.base_probs,
                           midbi.ext.ipsi.base_probs)), (
        "Ipsilateral base probabilities not the same."
    )
    assert np.all(np.equal(midbi.noext.trans_probs,
                           midbi.ext.trans_probs)), (
        "Transition probabilities not the same."
    )

    computed_ext_base_contra = (
        midbi.alpha_mix * midbi.noext.ipsi.base_probs
        + (1 - midbi.alpha_mix) * midbi.noext.contra.base_probs
    )
    assert np.all(np.isclose(computed_ext_base_contra,
                             midbi.ext.contra.base_probs)), (
        "Contralateral base probabilities for midline extension are wrong."
    )

def test_load_data(bisys, midbi, data, t_stages, modality_spsn):
    """Check that data gets loaded correctly. The mockup dataset for the midext
    case is designed such that only the contralateral side for patients with
    mid-sagittal tumor extension should have a different C-matrix and f-vector.
    """
    bisys.modalities = modality_spsn
    bisys.patient_data = data["bilateral"]

    midbi.modalities = modality_spsn
    midbi.patient_data = data["midext"]