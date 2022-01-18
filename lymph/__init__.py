"""This package contains code to model the spread of microscopic metastases
through a system of lymph node levels (LNLs), using either a Bayesian network
or a hidden Markov model."""

__description__ = "Package for statistical modelling of lymphatic metastatic spread."
__author__ = "Roman Ludwig"
__email__ = "roman.ludwig@usz.ch"
__uri__ = "https://github.com/rmnldwg/lymph"

# nopycln: file

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from .bilateral import Bilateral, BilateralSystem
from .edge import Edge
from .midline import MidlineBilateral
from .node import Node
from .unilateral import System, Unilateral
from .utils import EnsembleSampler, HDF5Mixin, change_base, system_from_hdf5

__all__ = [
    "Node",
    "Edge",
    "Unilateral", "System",
    "Bilateral", "BilateralSystem",
    "MidlineBilateral",
    "EnsembleSampler", "HDF5Mixin",
    "system_from_hdf5", "change_base",
]