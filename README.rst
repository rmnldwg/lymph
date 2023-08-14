.. image:: https://raw.githubusercontent.com/rmnldwg/lymph/main/docs/source/_static/github-social-card.png

.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/rmnldwg/lymph/blob/main/LICENSE
.. image:: https://img.shields.io/badge/GitHub-rmnldwg%2Flymph-blue.svg?style=flat
    :target: https://github.com/rmnldwg
.. image:: https://img.shields.io/badge/DOI-10.1038%2Fs41598--021--91544--1-success.svg?style=flat
    :target: https://doi.org/10.1038/s41598-021-91544-1
.. image:: https://github.com/rmnldwg/lymph/actions/workflows/tests.yml/badge.svg?style=flat
    :target: https://github.com/rmnldwg/lymph/actions
.. image:: https://github.com/rmnldwg/lymph/actions/workflows/build.yml/badge.svg?style=flat
    :target: https://pypi.org/project/lymph-model/
.. image:: https://codecov.io/gh/rmnldwg/lymph/branch/main/graph/badge.svg?token=LPXQPK5K78
    :target: https://codecov.io/gh/rmnldwg/lymph
.. image:: https://readthedocs.org/projects/lymph-model/badge
    :target: https://lymph-model.readthedocs.io


A Python package for statistical modelling of lymphatic metastatic spread in head & neck squamous cell carinoma (HNSCC).


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

If you use this code in you work you may either attribute use by citing our `paper <https://doi.org/10.1038/s41598-021-91544-1>`_ [8]_ or by using the ``CITATION.cff`` file in this repository, which specifically cites the software.


See also
========

The dataset(s)
--------------

A large and detailed dataset containing the patterns of lymphatic progression of patients treated at the USZ are available in `this repo <https://github.com/rmnldwg/lydata>`_ on GitHub. We have also published a paper on the dataset and our interface (described below) in *Radiotherapy & Oncology* [9]_. A preprint is available on *medRxiv* [10]_.

This data may be used to train the model.

LyProX interface
----------------

The above mentioned data can also be explored interactively using our online interface `LyProX <https://lyprox.org>`_ `(GitHub repo) <https://github.com/rmnldwg/lyprox>`_.






























References
==========

.. [1] Vincent Grégoire and Others, **Selection and delineation of lymph node target volumes in head and neck conformal radiotherapy. Proposal for standardizing terminology and procedure based on the surgical experience**, *Radiotherapy and Oncology*, vol. 56, pp. 135-150, 2000, doi: https://doi.org/10.1016/S0167-8140(00)00202-4.
.. [2] Vincent Grégoire, A. Eisbruch, M. Hamoir, and P. Levendag, **Proposal for the delineation of the nodal CTV in the node-positive and the post-operative neck**, *Radiotherapy and Oncology*, vol. 79, no. 1, pp. 15-20, Apr. 2006, doi: https://doi.org/10.1016/j.radonc.2006.03.009.
.. [3] Vincent Grégoire et al., **Delineation of the neck node levels for head and neck tumors: A 2013 update. DAHANCA, EORTC, HKNPCSG, NCIC CTG, NCRI, RTOG, TROG consensus guidelines**, *Radiotherapy and Oncology*, vol. 110, no. 1, pp. 172-181, Jan. 2014, doi: https://doi.org/10.1016/j.radonc.2013.10.010.
.. [4] Julian Biau et al., **Selection of lymph node target volumes for definitive head and neck radiation therapy: a 2019 Update**, *Radiotherapy and Oncology*, vol. 134, pp. 1-9, May 2019, doi: https://doi.org/10.1016/j.radonc.2019.01.018.
.. [5] Jatin. P. Shah, F. C. Candela, and A. K. Poddar, **The patterns of cervical lymph node metastases from squamous carcinoma of the oral cavity**, *Cancer*, vol. 66, no. 1, pp. 109-113, 1990, doi: https://doi.org/10.1002/1097-0142(19900701)66:1%3C109::AID-CNCR2820660120%3E3.0.CO;2-A.
.. [6] Laurence Bauwens et al., **Prevalence and distribution of cervical lymph node metastases in HPV-positive and HPV-negative oropharyngeal squamous cell carcinoma**, *Radiotherapy and Oncology*, vol. 157, pp. 122-129, Apr. 2021, doi: https://doi.org/10.1016/j.radonc.2021.01.028.
.. [7] Bertrand Pouymayou, P. Balermpas, O. Riesterer, M. Guckenberger, and J. Unkelbach, **A Bayesian network model of lymphatic tumor progression for personalized elective CTV definition in head and neck cancers**, *Physics in Medicine & Biology*, vol. 64, no. 16, p. 165003, Aug. 2019, doi: https://doi.org/10.1088/1361-6560/ab2a18.
.. [8] Roman Ludwig, B. Pouymayou, P. Balermpas, and J. Unkelbach, **A hidden Markov model for lymphatic tumor progression in the head and neck**, *Sci Rep*, vol. 11, no. 1, p. 12261, Dec. 2021, doi: https://doi.org/10.1038/s41598-021-91544-1.
.. [9] Roman Ludwig et al., **Detailed patient-individual reporting of lymph node involvement in oropharyngeal squamous cell carcinoma with an online interface**, *Radiotherapy and Oncology*, Feb. 2022, doi: https://doi.org/10.1016/j.radonc.2022.01.035.
.. [10] Roman Ludwig, J.-M. Hoffmann, B. Pouymayou et al., **Detailed patient-individual reporting of lymph node involvement in oropharyngeal squamous cell carcinoma with an online interface**, *medRxiv*, Dec. 2021. doi: https://doi.org/10.1101/2021.12.01.21267001.
