"""This package contains code to model the spread of microscopic metastases 
through a system of lymph node levels (LNLs), using either a Bayesian network 
or a hidden Markov model."""

__description__ = "Package for statistical modelling of lymphatic metastatic spread."
__author__ = "Roman Ludwig"
__email__ = "roman.ludwig@usz.ch"
__uri__ = "https://github.com/rmnldwg/lymph"

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import numpy as np
import scipy as sp 
import scipy.stats

from .node import *
from .edge import *
from .system import *

__all__ = [
    "Node",
    "Edge",
    "System",
    "BilateralSystem",
    "toStr",
]