"""
Make doctests in the lymph package discoverable by unittest.
"""
import doctest
import unittest

from lymph import diagnose_times, graph, helper, matrix, modalities
from lymph.models import bilateral, unilateral


def load_tests(loader, tests: unittest.TestSuite, ignore):
    """Load doctests from the lymph package."""
    tests.addTests(doctest.DocTestSuite(diagnose_times))
    tests.addTests(doctest.DocTestSuite(graph))
    tests.addTests(doctest.DocTestSuite(helper))
    tests.addTests(doctest.DocTestSuite(matrix))
    tests.addTests(doctest.DocTestSuite(modalities))

    tests.addTests(doctest.DocTestSuite(unilateral))
    tests.addTests(doctest.DocTestSuite(bilateral))
    return tests
