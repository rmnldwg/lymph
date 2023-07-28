"""Check functionality of the distribution over diagnose times."""
import unittest

import numpy as np
import scipy as sp

from lymph.descriptors.diagnose_times import Distribution, DistributionDict


class FixtureMixin:
    """Mixin that provides fixtures for the tests."""

    def setUp(self):
        self.max_time = 10
        self.array_arg = np.random.uniform(size=self.max_time + 1, low=0., high=10.)
        self.func_arg = lambda support, p: sp.stats.binom.pmf(support, self.max_time, p)


class DistributionTestCase(FixtureMixin, unittest.TestCase):
    """Test the distribution dictionary."""

    def test_frozen_distribution_without_max_time(self):
        """Test the creation of a frozen distribution without providing a max time."""
        dist = Distribution(self.array_arg)
        self.assertTrue(dist.is_frozen)
        self.assertFalse(dist.is_updateable)
        self.assertRaises(ValueError, dist.get_params)
        self.assertTrue(len(dist.support) == self.max_time + 1)
        self.assertTrue(len(dist.distribution) == self.max_time + 1)
        self.assertTrue(np.allclose(sum(dist.distribution), 1.))

    def test_frozen_distribution_with_max_time(self):
        """Test the creation of a frozen distribution where we provide the max_time."""
        dist = Distribution(self.array_arg)
        self.assertTrue(dist.is_frozen)
        self.assertFalse(dist.is_updateable)
        self.assertRaises(ValueError, dist.get_params)
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

        dist.set_params(0.5)
        self.assertTrue(len(dist.support) == self.max_time + 1)
        self.assertTrue(len(dist.distribution) == self.max_time + 1)
        self.assertTrue(dist.is_frozen)
        self.assertTrue(np.allclose(sum(dist.distribution), 1.))


class DistributionDictTestCase(FixtureMixin, unittest.TestCase):
    """Test the distribution dictionary."""

    def setUp(self):
        super().setUp()
        self.rng = np.random.default_rng(42)
        self.dist_dict = DistributionDict(max_time=self.max_time)

    def test_setitem_distribution_from_array(self):
        """Test setting a distribution created from an array."""
        self.dist_dict['test'] = Distribution(self.array_arg)
        self.assertTrue('test' in self.dist_dict)
        self.assertTrue(self.dist_dict.max_time == self.max_time)

    def test_setitem_distribution_from_func(self):
        """Test setting a distribution created from a function."""
        self.assertRaises(ValueError, Distribution, self.func_arg)
        self.dist_dict['test'] = Distribution(self.func_arg, max_time=self.max_time)
        self.assertTrue('test' in self.dist_dict)

    def test_setitem_from_array(self):
        """Test setting an item via an array distribution."""
        self.dist_dict['test'] = self.array_arg
        self.assertTrue('test' in self.dist_dict)

    def test_setitem_from_func(self):
        """Test setting an item via a parametrized distribution."""
        self.dist_dict['test'] = self.func_arg
        self.assertTrue('test' in self.dist_dict)

    def test_multiple_setitem(self):
        """Test setting multiple distributions."""
        for i in range(5):
            func = lambda support, p: sp.stats.binom.pmf(support, self.max_time, p)
            self.dist_dict[f"test_{i}"] = func

        self.assertTrue(len(self.dist_dict) == 5)
        for i in range(5):
            self.assertTrue(f"test_{i}" in self.dist_dict)
            self.assertTrue(self.dist_dict[f"test_{i}"].is_updateable)
            self.assertFalse(self.dist_dict[f"test_{i}"].is_frozen)
            param = self.rng.uniform()
            self.dist_dict[f"test_{i}"].set_params(param)
            returned_param, _ = self.dist_dict[f"test_{i}"].get_params()
            self.assertTrue(np.allclose(param, returned_param))
            self.assertTrue(self.dist_dict[f"test_{i}"].is_frozen)
