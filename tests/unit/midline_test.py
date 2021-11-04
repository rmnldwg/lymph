import pytest
import numpy as np
import scipy as sp
import pandas as pd
import lymph


@pytest.fixture
def midbi():
    graph = {('tumor', 'primary'): ['one', 'two'],
             ('lnl', 'one'):       ['two', 'three'],
             ('lnl', 'two'):       ['three'],
             ('lnl', 'three'):     []}
    return lymph.MidlineBilateral(graph=graph)


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
                           