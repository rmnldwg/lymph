"""
Test the Bayesian Unilateral Model.
"""
import unittest

import numpy as np
from binary_unilateral_test import ModelFixtureMixin


class BayesianUnilateralModelTestCase(ModelFixtureMixin, unittest.TestCase):
    """Test the Bayesian Unilateral Model."""

    def setUp(self):
        res = super().setUp()
        params_to_set = self.create_random_params(seed=123)
        self.model.assign_params(**params_to_set)
        return res

    def test_state_dist(self):
        """Test the state distribution."""
        bayes_state_dist = self.model.comp_state_dist(mode="BN")
        self.assertTrue(np.isclose(bayes_state_dist.sum(), 1.0))

    def test_obs_dist(self):
        """Test the observation distribution."""
        self.model.modalities = self.create_modalities()
        bayes_obs_dist = self.model.comp_obs_dist(mode="BN")
        self.assertTrue(np.isclose(bayes_obs_dist.sum(), 1.0))
