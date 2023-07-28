"""Check functionality of the distribution over diagnose times."""
import unittest

import numpy as np
import scipy as sp

from lymph.descriptors.diagnose_times import Distribution


class DistributionTestCase(unittest.TestCase):
    """Test the distribution dictionary."""

    def setUp(self):
        self.max_time = 10
        self.array_arg = np.random.uniform(size=self.max_time + 1, low=0., high=10.)
        self.func_arg = lambda support, p: sp.stats.binom.pmf(support, self.max_time, p)

    def test_frozen_distribution_without_max_time(self):
        """Test the creation of a frozen distribution without providing a max time."""
        dist = Distribution(self.array_arg)
        self.assertTrue(dist.is_frozen)
        self.assertFalse(dist.is_updateable)
        self.assertRaises(ValueError, dist.get_param)
        self.assertTrue(len(dist.support) == self.max_time + 1)
        self.assertTrue(len(dist.distribution) == self.max_time + 1)
        self.assertTrue(np.allclose(sum(dist.distribution), 1.))

    def test_frozen_distribution_with_max_time(self):
        """Test the creation of a frozen distribution where we provide the max_time."""
        dist = Distribution(self.array_arg)
        self.assertTrue(dist.is_frozen)
        self.assertFalse(dist.is_updateable)
        self.assertRaises(ValueError, dist.get_param)
        self.assertTrue(len(dist.support) == self.max_time + 1)
        self.assertTrue(len(dist.distribution) == self.max_time + 1)
        self.assertTrue(np.allclose(sum(dist.distribution), 1.))

        self.assertRaises(ValueError, Distribution, self.array_arg, max_time=5)

    def test_updateable_distribution_without_max_time(self):
        """Test the creation of an updateable distribution without providing a max time."""
        self.assertRaises(ValueError, Distribution, self.func_arg)

    def test_updateable_distribution_with_max_time(self):
        """Test the creation of an updateable distribution where we provide the max_time."""
        dist = Distribution(self.func_arg, max_time=self.max_time)
        self.assertFalse(dist.is_frozen)
        self.assertTrue(dist.is_updateable)
        self.assertRaises(ValueError, dist.get_param)

        dist.set_param(0.5)
        self.assertTrue(len(dist.support) == self.max_time + 1)
        self.assertTrue(len(dist.distribution) == self.max_time + 1)
        self.assertTrue(dist.is_frozen)
        self.assertTrue(np.allclose(sum(dist.distribution), 1.))
