"""Test that matrix generation caching is working correctly."""

import time
import unittest

import numpy as np
import scipy as sp

from . import fixtures


class CachingTestCase(
    fixtures.BinaryUnilateralModelMixin,
    unittest.TestCase,
):
    """Test that transition and observation matrices are cached."""

    def setUp(self):
        """Initialize the model."""
        super().setUp(graph_size="medium")
        # Set up distributions for the model
        self.model.set_distribution(
            "early",
            sp.stats.binom.pmf(
                np.arange(self.model.max_time + 1),
                self.model.max_time,
                0.4,
            ),
        )

    def test_transition_matrix_caching(self):
        """Check that transition matrix is cached and fast on repeated calls."""
        # Set some parameters
        self.model.set_params(**{"TtoII_spread": 0.8})

        # First call - should compute
        start = time.time()
        tm1 = self.model.transition_matrix()
        time1 = time.time() - start

        # Second call - should be cached
        start = time.time()
        tm2 = self.model.transition_matrix()
        time2 = time.time() - start

        # Third call - should be cached
        start = time.time()
        tm3 = self.model.transition_matrix()
        time3 = time.time() - start

        # Check that cached calls are faster than the first call
        self.assertLess(time2, time1, "Second call should be faster (cached)")
        self.assertLess(time3, time1, "Third call should be faster (cached)")

        # Check that all three return the same object in memory
        self.assertIs(tm1, tm2, "Cached calls should return same object")
        self.assertIs(tm2, tm3, "Cached calls should return same object")

        # Verify arrays are actually equal
        np.testing.assert_array_equal(tm1, tm2)
        np.testing.assert_array_equal(tm2, tm3)

    def test_transition_matrix_changes_with_params(self):
        """Check that different parameters produce different matrices."""
        # Set initial parameters
        self.model.set_params(**{"TtoII_spread": 0.8})
        tm1 = self.model.transition_matrix()

        # Change parameters
        self.model.set_params(**{"TtoII_spread": 0.5})
        tm2 = self.model.transition_matrix()

        # Matrices should be different
        self.assertIsNot(
            tm1,
            tm2,
            "Different params should produce different cached matrices",
        )
        self.assertFalse(
            np.array_equal(tm1, tm2),
            "Different param values should produce different transition matrices",
        )

    def test_observation_matrix_caching(self):
        """Check that observation matrix is cached and fast on repeated calls."""
        # First call - should compute
        start = time.time()
        om1 = self.model.observation_matrix()
        time1 = time.time() - start

        # Second call - should be cached
        start = time.time()
        om2 = self.model.observation_matrix()
        time2 = time.time() - start

        # Third call - should be cached
        start = time.time()
        om3 = self.model.observation_matrix()
        time3 = time.time() - start

        # Check that cached calls are faster than the first call
        self.assertLess(time2, time1, "Second call should be faster (cached)")
        self.assertLess(time3, time1, "Third call should be faster (cached)")

        # Check that all three return the same object in memory
        self.assertIs(om1, om2, "Cached calls should return same object")
        self.assertIs(om2, om3, "Cached calls should return same object")

        # Verify arrays are actually equal
        np.testing.assert_array_equal(om1, om2)
        np.testing.assert_array_equal(om2, om3)
