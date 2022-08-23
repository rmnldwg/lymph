"""
This package contains code to model the spread of microscopic metastases
through a system of lymph node levels (LNLs), using either a Bayesian network
or a hidden Markov model.
"""

from ._version import version

__version__ = version
__description__ = "Package for statistical modelling of lymphatic metastatic spread."
__author__ = "Roman Ludwig"
__email__ = "roman.ludwig@usz.ch"
__uri__ = "https://github.com/rmnldwg/lymph"

# nopycln: file

from .bilateral import Bilateral, BilateralSystem
from .edge import Edge
from .midline import MidlineBilateral
from .node import Node
from .timemarg import Marginalizor, MarginalizorDict
from .unilateral import System, Unilateral

__all__ = [
    "Node",
    "Edge",
    "Unilateral", "System",
    "Bilateral", "BilateralSystem",
    "MidlineBilateral",
    "Marginalizor", "MarginalizorDict",
]