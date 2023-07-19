"""Test the unilateral module."""
from typing import Dict
import unittest

import numpy as np

from lymph.models import Unilateral
from lymph.graph import LymphNodeLevel, Tumor


class BinaryFixtureMixin:
    """Mixin class for simple binary model fixture creation."""

    def setUp(self):
        """Initialize a simple binary model."""
        self.graph = {
            ("tumor", "T"): ["I", "II", "III", "IV"],
            ("lnl", "I"): [],
            ("lnl", "II"): ["I", "III"],
            ("lnl", "III"): ["IV"],
            ("lnl", "IV"): [],
        }
        self.model = Unilateral(graph=self.graph)

    def create_random_params(self, seed: int = 42) -> Dict[str, float]:
        """Create random parameters for the model."""
        rng = np.random.default_rng(seed)
        return {name: rng.random() for name in self.model.edge_params.keys()}


class InitBinaryTestCase(BinaryFixtureMixin, unittest.TestCase):
    """Test the initialization of a binary model."""

    def test_num_nodes(self):
        """Check number of nodes initialized."""
        self.assertEqual(len(self.model.nodes), 5)
        self.assertEqual(len(self.model.tumors), 1)
        self.assertEqual(len(self.model.lnls), 4)

    def test_num_edges(self):
        """Check number of edges initialized."""
        self.assertEqual(len(self.model.edges), 7)
        self.assertEqual(len(self.model.tumor_edges), 4)
        self.assertEqual(len(self.model.lnl_edges), 3)
        self.assertEqual(len(self.model.growth_edges), 0)

    def test_tumor(self):
        """Make sure the tumor has been initialized correctly."""
        tumor = self.model.find_node("T")
        state = tumor.state
        self.assertIsInstance(tumor, Tumor)
        self.assertListEqual(tumor.allowed_states, [state])

    def test_lnls(self):
        """Test they are all binary lymph node levels."""
        for lnl in self.model.lnls:
            self.assertIsInstance(lnl, LymphNodeLevel)
            self.assertTrue(lnl.is_binary)

    def test_tumor_to_lnl_edges(self):
        """Make sure the tumor to LNL edges have been initialized correctly."""
        tumor = self.model.find_node("T")
        receiving_lnls = self.graph[("tumor", "T")]
        connecting_edge_names = [f"{tumor.name}_to_{lnl}" for lnl in receiving_lnls]

        for edge in self.model.tumor_edges:
            self.assertEqual(edge.start.name, "T")
            self.assertIn(edge.end.name, receiving_lnls)
            self.assertTrue(edge.is_tumor_spread)
            self.assertIn(edge.name, connecting_edge_names)

    def test_lnl_to_lnl_edges(self):
        """Make sure the LNL to LNL edges have been initialized correctly."""
        for edge in self.model.lnl_edges:
            self.assertIsInstance(edge.start, LymphNodeLevel)
            self.assertIsInstance(edge.end, LymphNodeLevel)
            self.assertIn(edge.start.name, ["II", "III"])
            self.assertIn(edge.end.name, ["I", "III", "IV"])


class BinaryParameterAssignmentTestCase(BinaryFixtureMixin, unittest.TestCase):
    """Test the assignment of parameters in a binary model."""

    def test_edge_params_assignment_via_lookup(self):
        """Make sure the spread parameters are assigned correctly."""
        params_to_set = self.create_random_params(seed=42)
        for name, value in params_to_set.items():
            self.model.edge_params[name].set(value)
            self.assertEqual(self.model.edge_params[name].get(), value)

    def test_edge_params_assignment_via_method(self):
        """Make sure the spread parameters are assigned correctly."""
        params_to_set = self.create_random_params(seed=43)
        self.model.assign_parameters(**params_to_set)
        for name, value in params_to_set.items():
            self.assertEqual(self.model.edge_params[name].get(), value)

    def test_direct_assignment_raises_error(self):
        """Make sure direct assignment of parameters raises an error."""
        with self.assertRaises(TypeError):
            self.model.edge_params["spread_T_to_I"] = 0.5

    def test_transition_matrix_deletion(self):
        """Check if the transition matrix gets deleted when a parameter is set.

        NOTE: This test is disabled because apparently, the `model` instance is
        changed during the test and the `_transition_matrix` attribute is deleted on
        the wrong instance. I have no clue why, but generally, the method works.
        """
        _ = self.model.transition_matrix
        self.assertTrue(hasattr(self.model, "_transition_matrix"))
        self.model.edge_params["spread_T_to_I"].set(0.5)
        self.assertFalse(hasattr(self.model, "_transition_matrix"))


class BinaryTransitionMatrixTestCase(BinaryFixtureMixin, unittest.TestCase):
    """Test the generation of the transition matrix in a binary model."""

    def setUp(self):
        """Initialize a simple binary model."""
        self.graph = {
            ("tumor", "T"): ["II", "III"],
            ("lnl", "II"): ["III"],
            ("lnl", "III"): [],
        }
        self.model = Unilateral(graph=self.graph)

        params_to_set = self.create_random_params(seed=42)
        self.model.assign_parameters(**params_to_set)

    def test_transition_matrix_shape(self):
        """Make sure the transition matrix has the correct shape."""
        self.assertEqual(self.model.transition_matrix.shape, (4, 4))

    def test_transition_matrix_row_sums(self):
        """Make sure the rows of the transition matrix sum to one."""
        row_sums = np.sum(self.model.transition_matrix, axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))


if __name__ == "__main__":
    unittest.main()
