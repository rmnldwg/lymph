"""Check functionality of the distribution over diagnose times."""
import unittest
import warnings

import numpy as np
import scipy as sp

from lymph.diagnose_times import Distribution, DistributionsUserDict


class FixtureMixin:
    """Mixin that provides fixtures for the tests."""

    @staticmethod
    def binom_pmf(
        support: np.ndarray,
        max_time: int = 10,
        p: float = 0.5,
    ) -> np.ndarray:
        """Binomial probability mass function."""
        if max_time <= 0:
            raise ValueError("max_time must be a positive integer.")
        if len(support) != max_time + 1:
            raise ValueError("support must have length max_time + 1.")
        if not 0. <= p <= 1.:
            raise ValueError("p must be between 0 and 1.")

        return sp.stats.binom.pmf(support, max_time, p)


    def setUp(self):
        self.max_time = 10
        self.array_arg = np.random.uniform(size=self.max_time + 1, low=0., high=10.)
        self.func_arg = lambda support, p=0.5: self.binom_pmf(support, self.max_time, p)


class DistributionTestCase(FixtureMixin, unittest.TestCase):
    """Test the distribution dictionary."""

    def test_frozen_distribution_without_max_time(self):
        """Test the creation of a frozen distribution without providing a max time."""
        dist = Distribution(self.array_arg)
        self.assertFalse(dist.is_updateable)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.assertEqual({}, dist.get_params(as_dict=True))
        self.assertTrue(len(dist.support) == self.max_time + 1)
        self.assertTrue(len(dist.distribution) == self.max_time + 1)
        self.assertTrue(np.allclose(sum(dist.distribution), 1.))

    def test_frozen_distribution_with_max_time(self):
        """Test the creation of a frozen distribution where we provide the max_time."""
        dist = Distribution(self.array_arg)
        self.assertFalse(dist.is_updateable)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.assertEqual({}, dist.get_params(as_dict=True))
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
        self.assertTrue(dist.is_updateable)

        dist.set_params(p=0.5)
        self.assertTrue(len(dist.support) == self.max_time + 1)
        self.assertTrue(len(dist.distribution) == self.max_time + 1)
        self.assertTrue(np.allclose(sum(dist.distribution), 1.))

    def test_updateable_distribution_raises_value_error(self):
        """Check that an invalid parameter raises a ValueError."""
        dist = Distribution(self.func_arg, max_time=self.max_time)
        self.assertTrue(dist.is_updateable)
        self.assertRaises(ValueError, dist.set_params, p=1.5)


class DistributionDictTestCase(FixtureMixin, unittest.TestCase):
    """Test the distribution dictionary."""

    def setUp(self):
        super().setUp()
        self.rng = np.random.default_rng(42)
        self.dist_dict = DistributionsUserDict(
            max_time=self.max_time,
            trigger_callbacks=[],
        )

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
            func = lambda support, p=0.2: sp.stats.binom.pmf(support, self.max_time, p)
            self.dist_dict[f"test_{i}"] = func

        self.assertTrue(len(self.dist_dict) == 5)
        for i in range(5):
            self.assertTrue(f"test_{i}" in self.dist_dict)
            self.assertTrue(self.dist_dict[f"test_{i}"].is_updateable)
            param = self.rng.uniform()
            self.dist_dict[f"test_{i}"].set_params(p=param)
            returned_param = self.dist_dict[f"test_{i}"].get_params(as_dict=True)
            self.assertTrue(np.allclose(param, returned_param["p"]))
