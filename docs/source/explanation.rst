What is ``lymph``?
==================

``lymph`` is a Python package for statistical modelling of lymphatic metastatic spread in head & neck squamous cell carinoma (HNSCC).


Motivation
==========

HNSCC spreads though the lymphatic system of the neck and forms metastases in regional lymph nodes. Macroscopic metastases can be detected with imaging modalities like MRI, PET and CT scans. They are then consequently included in the target volume, when radiotherapy is chosen as part of the treatment. However, microscopic metastases are too small be diagnosed with current imaging techniques.

To account for this microscopic involvement, parts of the lymphatic system are often irradiated electively to increase tumor control. Which parts are included in this elective clinical target volume is currently decided based on guidelines [1]_ [2]_ [3]_ [4]_. These in turn are derived from reports of the prevalence of involvement per lymph node level (LNL), i.e. the portion of patients that were diagnosed with metastases in any given LNL, stratified by primary tumor location. It is recommended to include a LNL in the elective target volume if 10 - 15% of patients showed involvement in that particular level.

However, while the prevalence of involvement has been reported in the literature [5]_ [6]_, and the general lymph drainage pathways are understood well, the detailed progression patterns of HNSCC remain poorly quantified. We believe that the risk for microscopic involvement in an LNL depends highly on the specific diagnose of a particular patient and their treatment can hence be personalized if the progression patterns were better quantified.


Our Goal
========

With this Python package we want to provide a framework to accurately predict the risk for microscopic metastases in any lymph node level for the specific diagnose a particular patient presents with.

The implemented model is highly interpretable and was developed together with clinicians to accurately represent the anatomy of the lymphatic drainiage. It can be trained with data that reports the patterns of lymphatic progression in detail, like the `dataset(s) <https://github.com/rmnldwg/lydata>`_ we collected at our institution, the University Hospital Zurich (USZ).

The mathematical details of the models can be found in in our earlier publications [7]_ [8]_.


Get started
===========

To learn how to use this package, head over to our `documentation <https://lymph-model.readthedocs.io>`_ where we explain the API of the package and also provide a `quickstart guide <https://lymph-model.readthedocs.io/en/latest/quickstart.html>`_.

The implementation is pure-python and has only a few dependencies. However, it is intended to be used with powerful inference algorithms, e.g. the great sampling package `emcee <https://github.com/dfm/emcee>`_, which we used for our results.


Attribution
===========

If you use this code in you work you may either attribute use by citing our `paper <https://doi.org/10.1038/s41598-021-91544-1>`_ or by using the ``CITATION.cff`` file in this repository, which specifically cites the software.


See also
========

The dataset(s)
--------------

A large and detailed dataset containing the patterns of lymphatic progression of patients treated at the USZ are available `here <https://github.com/rmnldwg/lydata>`_.

This data may be used to train the model.

LyProX interface
----------------

The above mentioned data can also be explored interactively using our online interface `LyProX <https://lyprox.org>`_ `(GitHub repo) <https://github.com/rmnldwg/lyprox>`_.
