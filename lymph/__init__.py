"""The lymph package implements models for lymphatic metastatic spread.

It forms the basis of the first comprehensive statistical model to predict personalized
risks for occult disease in head and neck squamous cell carcinoma patients.

These models may at some point be used to inform the elective clinical target volume
definition in radiotherapy treatment planning.
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

from lymph import diagnosis_times, graph, matrix, models
from lymph.utils import clinical, pathological

__all__ = [
    "diagnosis_times",
    "matrix",
    "graph",
    "models",
    "clinical",
    "pathological",
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
