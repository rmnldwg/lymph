"""
This package contains code to model the spread of microscopic metastases
through a system of lymph node levels (LNLs), using either a Bayesian network
or a hidden Markov model.
"""
import logging
from logging import NullHandler, StreamHandler

from lymph._version import version

__version__ = version
__description__ = "Package for statistical modelling of lymphatic metastatic spread."
__author__ = "Roman Ludwig"
__email__ = "roman.ludwig@usz.ch"
__uri__ = "https://github.com/rmnldwg/lymph"

# nopycln: file

from lymph import diagnose_times, graph, matrix, models
from lymph.helper import clinical, pathological

__all__ = [
    "diagnose_times", "matrix",
    "graph", "models",
    "clinical", "pathological",
]


# configure library logging akin to how it was done in urllib3 v2.0.4:
# https://github.com/urllib3/urllib3/blob/2.0.4/src/urllib3/__init__.py#L87-L107
logging.getLogger(__name__).addHandler(NullHandler())

def add_stderr_logging(level: int = logging.DEBUG) -> StreamHandler:
    """Add a stderr log handler to the logger."""
    logger = logging.getLogger(__name__)
    handler = StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug("Added stderr logging with level %s.", level)
    return handler
