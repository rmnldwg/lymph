.. module: components

.. _components:


Components
===========

Below we document the core components implementing the Bayesian network and hidden
Markov model we use to compute probabilistic predictions of lymphatic tumor spread.


Diagnostic Modalities
---------------------

This module allows the user to define diagnostic modalities and their sensitivity/specificity
values. This is necessary to compute the likelihood of a dataset (that was created by
recoding the output of diagnostic modalities), given the model and its parameters (which
we want to learn).

.. automodule:: lymph.modalities
    :members:
    :special-members: __init__
    :show-inheritance:


Marginalization over Diagnose Times
-----------------------------------

The hidden Markov model we implement here assumes that every patient started off with
a healthy neck, meaning no lymph node levels harboured any metastases. This is a valid
assumption, but brings with it the issue of determining *how long ago* this likely was.

This module allows the user to define a distribution over abstract time-steps that
indicate for different T-categories how probable a diagnosis at this time-step was.
That allows us to treat T1 and T4 patients fundamentally in the same way, even with the
same parameters, except for the parametrization of their respective distribution over
the time of diagnosis.

.. automodule:: lymph.diagnose_times
    :members:
    :special-members: __init__
    :show-inheritance:

Matrices
--------

In this module we implement the core matrices, like the transition or the observation
matrix.

.. automodule:: lymph.matrix
    :members:
    :special-members: __init__
    :show-inheritance:
