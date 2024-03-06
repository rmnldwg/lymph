"""
Test the Bayesian Unilateral Model.
"""
import numpy as np

from . import fixtures


class BayesianUnilateralModelTestCase(
    fixtures.BinaryUnilateralModelMixin,
    fixtures.IgnoreWarningsTestCase,
):
    """Test the Bayesian Unilateral Model."""

    def setUp(self):
        super().setUp()
        self.model.set_params(**self.create_random_params())
        self.model.replace_all_modalities(fixtures.MODALITIES)
        self.load_patient_data(filename="2021-usz-oropharynx.csv")

    def test_state_dist(self):
        """Test the state distribution."""
        bayes_state_dist = self.model.state_dist(mode="BN")
        self.assertTrue(np.isclose(bayes_state_dist.sum(), 1.0))

    def test_obs_dist(self):
        """Test the observation distribution."""
        bayes_obs_dist = self.model.obs_dist(mode="BN")
        self.assertTrue(np.isclose(bayes_obs_dist.sum(), 1.0))

    def test_log_likelihood_smaller_zero(self):
        """Test the likelihood."""
        likelihood = self.model.likelihood(mode="BN")
        self.assertLessEqual(likelihood, 0.)

    def test_likelihood_invalid_params_isinf(self):
        """Make sure the likelihood is `-np.inf` for invalid parameters."""
        random_params = self.create_random_params()
        for name in random_params:
            random_params[name] += 1.
        likelihood = self.model.likelihood(
            given_params=random_params,
            log=True,
            mode="BN",
        )
        self.assertEqual(likelihood, -np.inf)
