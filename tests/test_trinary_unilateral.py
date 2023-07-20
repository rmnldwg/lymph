"""Test the trinary unilateral system."""
import unittest
from typing import Dict

import numpy as np

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
        self.model = Unilateral(graph=self.graph, allowed_states=[0,1,2])

    def create_random_params(self, seed: int = 42) -> Dict[str, float]:
        """Create random parameters for the model."""
        rng = np.random.default_rng(seed)
        return {name: rng.random() for name in self.model.edge_params.keys()}


class TrinaryTransitionMatrixTestCase(TrinaryFixtureMixin, unittest.TestCase):
    """Test the transition matrix of a trinary model."""

    def setUp(self):
        super().setUp()
        params_to_set = self.create_random_params(seed=123)
        self.model.assign_parameters(**params_to_set)

    def test_edge_transition_tensors(self) -> None:
        """Test the tensors associated with each edge.

        NOTE: I am using this only in debug mode to look a the tensors. I am not sure
        how to test them yet.
        """
        base_edge_tensor = self.model.tumor_edges[0].comp_transition_tensor()
        lnl_edge_tensor = self.model.lnl_edges[0].comp_transition_tensor()
        growth_edge_tensor = self.model.growth_edges[0].comp_transition_tensor()
        self.assertTrue(True)
