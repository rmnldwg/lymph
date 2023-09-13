"""
Test the Bayesian Unilateral Model.
"""
import unittest

import numpy as np

from tests import fixtures


class BayesianUnilateralModelTestCase(fixtures.BinaryUnilateralModelMixin, unittest.TestCase):
    """Test the Bayesian Unilateral Model."""

    def setUp(self):
        super().setUp()
        self.model.assign_params(**self.create_random_params())

    def test_state_dist(self):
        """Test the state distribution."""
        bayes_state_dist = self.model.comp_state_dist(mode="BN")
        self.assertTrue(np.isclose(bayes_state_dist.sum(), 1.0))

    def test_obs_dist(self):
        """Test the observation distribution."""
        self.model.modalities = fixtures.MODALITIES
        bayes_obs_dist = self.model.comp_obs_dist(mode="BN")
        self.assertTrue(np.isclose(bayes_obs_dist.sum(), 1.0))
