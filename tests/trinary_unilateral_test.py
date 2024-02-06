"""Test the trinary unilateral system."""
import unittest

import fixtures
import numpy as np
import pandas as pd

from lymph.graph import LymphNodeLevel


class TrinaryInitTestCase(fixtures.TrinaryFixtureMixin, unittest.TestCase):
    """Testing the basic initialization of a trinary model."""

    def test_is_trinary(self) -> None:
        """Test if the model is trinary."""
        self.assertTrue(self.model.is_trinary)

    def test_lnls(self):
        """Test they are all trinary lymph node levels."""
        model_allowed_states = self.model.graph.allowed_states
        self.assertEqual(len(model_allowed_states), 3)

        for lnl in self.model.graph.lnls.values():
            self.assertIsInstance(lnl, LymphNodeLevel)
            self.assertTrue(lnl.is_trinary)
            self.assertEqual(lnl.allowed_states, model_allowed_states)


class TrinaryTransitionMatrixTestCase(fixtures.TrinaryFixtureMixin, unittest.TestCase):
    """Test the transition matrix of a trinary model."""

    def setUp(self):
        super().setUp()
        params_to_set = self.create_random_params()
        self.model.assign_params(**params_to_set)

    def test_edge_transition_tensors(self) -> None:
        """Test the tensors associated with each edge.

        NOTE: I am using this only in debug mode to look a the tensors. I am not sure
        how to test them yet.
        """
        base_edge_tensor = list(self.model.graph.tumor_edges.values())[0].transition_tensor
        row_sums = base_edge_tensor.sum(axis=2)
        self.assertTrue(np.allclose(row_sums, 1.0))

        lnl_edge_tensor = list(self.model.graph.lnl_edges.values())[0].transition_tensor
        row_sums = lnl_edge_tensor.sum(axis=2)
        self.assertTrue(np.allclose(row_sums, 1.0))

        growth_edge_tensor = list(self.model.graph.growth_edges.values())[0].transition_tensor
        row_sums = growth_edge_tensor.sum(axis=2)
        self.assertTrue(np.allclose(row_sums, 1.0))

    def test_transition_matrix(self) -> None:
        """Test the transition matrix of the model."""
        transition_matrix = self.model.transition_matrix()
        row_sums = transition_matrix.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))


class TrinaryObservationMatrixTestCase(fixtures.TrinaryFixtureMixin, unittest.TestCase):
    """Test the observation matrix of a trinary model."""

    def setUp(self):
        super().setUp()
        self.model.modalities = self.get_modalities_subset(
            names=["diagnostic_consensus", "pathology"],
        )

    def test_observation_matrix(self) -> None:
        """Test the observation matrix of the model."""
        num_lnls = len(self.model.graph.lnls)
        num = num_lnls * len(self.model.modalities)
        observation_matrix = self.model.observation_matrix()
        self.assertEqual(observation_matrix.shape, (3 ** num_lnls, 2 ** num))

        row_sums = observation_matrix.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))


class TrinaryDiagnoseMatricesTestCase(fixtures.TrinaryFixtureMixin, unittest.TestCase):
    """Test the diagnose matrix of a trinary model."""

    def setUp(self):
        super().setUp()
        self.model.modalities = fixtures.MODALITIES
        self.load_patient_data(filename="2021-usz-oropharynx.csv")

    def get_patient_data(self) -> pd.DataFrame:
        """Load an example dataset that has both clinical and pathology data."""
        return pd.read_csv("tests/data/2021-usz-oropharynx.csv", header=[0, 1, 2])

    def test_diagnose_matrices_shape(self) -> None:
        """Test the diagnose matrix of the model."""
        for t_stage in ["early", "late"]:
            num_lnls = len(self.model.graph.lnls)
            num_patients = (self.model.patient_data["_model", "#", "t_stage"] == t_stage).sum()
            diagnose_matrix = self.model.diagnose_matrices[t_stage]
            self.assertEqual(diagnose_matrix.shape, (3 ** num_lnls, num_patients))


class TrinaryLikelihoodTestCase(fixtures.TrinaryFixtureMixin, unittest.TestCase):
    """Test the likelihood of a trinary model."""

    def setUp(self):
        """Load patient data."""
        super().setUp()
        self.model.modalities = fixtures.MODALITIES
        self.init_diag_time_dists(early="frozen", late="parametric")
        self.model.assign_params(**self.create_random_params())
        self.load_patient_data(filename="2021-usz-oropharynx.csv")

    def test_log_likelihood_smaller_zero(self):
        """Make sure the log-likelihood is smaller than zero."""
        likelihood = self.model.likelihood(log=True, mode="HMM")
        self.assertLess(likelihood, 0.)

    def test_likelihood_invalid_params_isinf(self):
        """Make sure the likelihood is `-np.inf` for invalid parameters."""
        random_params = self.create_random_params()
        for name in random_params:
            random_params[name] += 1.
        likelihood = self.model.likelihood(
            given_param_kwargs=random_params,
            log=True,
            mode="HMM",
        )
        self.assertEqual(likelihood, -np.inf)


class TrinaryRiskTestCase(fixtures.TrinaryFixtureMixin, unittest.TestCase):
    """Test the risk of a trinary model."""

    def setUp(self):
        """Load patient data."""
        super().setUp()
        self.model.modalities = fixtures.MODALITIES
        self.init_diag_time_dists(early="frozen", late="parametric")
        self.load_patient_data(filename="2021-usz-oropharynx.csv")

    def create_random_diagnoses(self):
        """Create a random diagnosis for each modality and LNL."""
        lnl_names = list(self.model.graph.lnls.keys())
        diagnoses = {}

        for modality in self.model.modalities:
            diagnoses[modality] = fixtures.create_random_pattern(lnl_names)

        return diagnoses

    def test_risk_is_probability(self):
        """Make sure the risk is a probability."""
        risk = self.model.risk(
            involvement=fixtures.create_random_pattern(lnls=list(self.model.graph.lnls.keys())),
            given_diagnoses=self.create_random_diagnoses(),
            given_param_kwargs=self.create_random_params(),
            t_stage=self.rng.choice(["early", "late"]),
        )
        self.assertGreaterEqual(risk, 0.)
        self.assertLessEqual(risk, 1.)
