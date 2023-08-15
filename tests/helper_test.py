"""
Test helper functions mainly by executing their doctests.
"""
import doctest

from lymph import helper


def load_tests(loader, tests, ignore):
    """Add doctests to the unittests."""
    tests.addTests(doctest.DocTestSuite(helper))
    return tests
