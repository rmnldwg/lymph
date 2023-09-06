"""Test the trinary unilateral system."""
import unittest

import numpy as np
import pandas as pd

from lymph.descriptors.modalities import Clinical, Modality, Pathological
from lymph.models import Unilateral


class TrinaryFixtureMixin:
    """Mixin class for simple trinary model fixture creation."""

    def setUp(self):
        """Initialize a simple trinary model."""
        self.graph = {
            ("tumor", "T"): ["I", "II", "III", "IV"],
            ("lnl", "I"): [],
            ("lnl", "II"): ["I", "III"],
            ("lnl", "III"): ["IV"],
            ("lnl", "IV"): [],
        }
        self.model = Unilateral(graph_dict=self.graph, allowed_states=[0,1,2])

    def create_random_params(self, seed: int = 42) -> dict[str, float]:
        """Create random parameters for the model."""
        rng = np.random.default_rng(seed)
        params = {
            f"{name}_{type_}": rng.random()
            for name, edge in self.model.graph.edges.items()
            for type_ in edge.get_params(as_dict=True).keys()
        }
        params.update({
            f"{t_stage}_{type_}": rng.random()
            for t_stage, dist in self.model.diag_time_dists.items()
            for type_ in dist.get_params(as_dict=True).keys()
        })
        return params

    def get_modalities_subset(self, names: list[str]) -> dict[str, Modality]:
        """Create a dictionary of modalities."""
        modalities_in_data = {
            "CT": Clinical(specificity=0.76, sensitivity=0.81),
            "MRI": Clinical(specificity=0.63, sensitivity=0.81),
            "PET": Clinical(specificity=0.86, sensitivity=0.79),
            "FNA": Pathological(specificity=0.98, sensitivity=0.80),
            "diagnostic_consensus": Clinical(specificity=0.86, sensitivity=0.81),
            "pathology": Pathological(specificity=1.0, sensitivity=1.0),
            "pCT": Clinical(specificity=0.86, sensitivity=0.81),
        }
        return {name: modalities_in_data[name] for name in names}


class TrinaryTransitionMatrixTestCase(TrinaryFixtureMixin, unittest.TestCase):
    """Test the transition matrix of a trinary model."""

    def setUp(self):
        super().setUp()
        params_to_set = self.create_random_params(seed=123)
        self.model.assign_params(**params_to_set)

    def test_edge_transition_tensors(self) -> None:
        """Test the tensors associated with each edge.

        NOTE: I am using this only in debug mode to look a the tensors. I am not sure
        how to test them yet.
        """
        base_edge_tensor = list(self.model.graph.tumor_edges.values())[0].comp_transition_tensor()
        row_sums = base_edge_tensor.sum(axis=2)
        self.assertTrue(np.allclose(row_sums, 1.0))

        lnl_edge_tensor = list(self.model.graph.lnl_edges.values())[0].comp_transition_tensor()
        row_sums = lnl_edge_tensor.sum(axis=2)
        self.assertTrue(np.allclose(row_sums, 1.0))

        growth_edge_tensor = list(self.model.graph.growth_edges.values())[0].comp_transition_tensor()
        row_sums = growth_edge_tensor.sum(axis=2)
        self.assertTrue(np.allclose(row_sums, 1.0))

    def test_transition_matrix(self) -> None:
        """Test the transition matrix of the model."""
        transition_matrix = self.model.transition_matrix
        row_sums = transition_matrix.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))


class TrinaryObservationMatrixTestCase(TrinaryFixtureMixin, unittest.TestCase):
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
        observation_matrix = self.model.observation_matrix
        self.assertEqual(observation_matrix.shape, (3 ** num_lnls, 2 ** num))

        row_sums = observation_matrix.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))


class TrinaryDiagnoseMatricesTestCase(TrinaryFixtureMixin, unittest.TestCase):
    """Test the diagnose matrix of a trinary model."""

    def setUp(self):
        super().setUp()
        self.model.load_patient_data(self.get_patient_data(), side="ipsi")
        _ = self.model.diagnose_matrices

    def get_patient_data(self) -> pd.DataFrame:
        """Load an example dataset that has both clinical and pathology data."""
        return pd.read_csv("tests/data/2021-clb-oropharynx.csv", header=[0, 1, 2])

    def test_diagnose_matrices_shape(self) -> None:
        """Test the diagnose matrix of the model."""
        for t_stage in ["early", "late"]:
            num_lnls = len(self.model.graph.lnls)
            num_patients = (self.model.patient_data["_model", "#", "t_stage"] == t_stage).sum()
            diagnose_matrix = self.model.diagnose_matrices[t_stage]
            self.assertEqual(diagnose_matrix.shape, (3 ** num_lnls, num_patients))
