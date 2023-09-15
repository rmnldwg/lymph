"""
Full integration test directly taken from the quickstart guide. Aimed at checking the
computed value of the likelihood function.
"""
import unittest

import fixtures
import numpy as np
import scipy as sp

import lymph


def late_binomial(support: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Parametrized binomial distribution."""
    return sp.stats.binom.pmf(support, n=support[-1], p=p)


class IntegrationTestCase(fixtures.BinaryUnilateralModelMixin, unittest.TestCase):
    """Run a stripped down version of the quickstart guide."""

    def setUp(self):
        self.graph_dict = fixtures.get_graph(size="medium")
        self.model = lymph.models.Unilateral.binary(graph_dict=self.graph_dict)
        self.model.modalities = {"PET": [0.86, 0.79]}
        self.load_patient_data("2021-usz-oropharynx.csv")

        early_fixed = sp.stats.binom.pmf(
            np.arange(self.model.max_time + 1),
            self.model.max_time,
            0.4,
        )
        self.model.diag_time_dists["early"] = early_fixed
        self.model.diag_time_dists["late"] = late_binomial

    def test_likelihood_value(self):
        """Check that the computed likelihood is correct."""
        test_probabilities = [0.02, 0.24, 0.03, 0.2, 0.23, 0.18, 0.18, 0.5]
        llh = self.model.likelihood(given_param_args=test_probabilities, log=True)
        self.assertAlmostEqual(llh, -586.8723971388224, places=10)
