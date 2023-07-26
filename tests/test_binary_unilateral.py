"""Test the binary unilateral system."""
import unittest
from typing import Dict

import numpy as np

from lymph.graph import LymphNodeLevel, Tumor
from lymph.models import Unilateral


class FixtureMixin:
    """Mixin class for simple binary model fixture creation."""

    def setUp(self):
        """Initialize a simple binary model."""
        large_graph = {
            ("tumor", "T"): ["I", "II", "III", "IV", "V", "VII"],
            ("lnl", "I"): [],
            ("lnl", "II"): ["I", "III", "V"],
            ("lnl", "III"): ["IV", "V"],
            ("lnl", "IV"): [],
            ("lnl", "V"): [],
            ("lnl", "VII"): [],
        }
        medium_graph = {
            ("tumor", "T"): ["II", "III", "V"],
            ("lnl", "II"): ["III", "V"],
            ("lnl", "III"): ["V"],
            ("lnl", "V"): [],
        }
        small_graph = {
            ("tumor", "T"): ["II", "III"],
            ("lnl", "II"): ["III"],
            ("lnl", "III"): [],
        }
        self.graph = large_graph
        self.model = Unilateral(graph=self.graph)

    def create_random_params(self, seed: int = 42) -> Dict[str, float]:
        """Create random parameters for the model."""
        rng = np.random.default_rng(seed)
        return {name: rng.random() for name in self.model.edge_params.keys()}


class InitTestCase(FixtureMixin, unittest.TestCase):
    """Test the initialization of a binary model."""

    def test_num_nodes(self):
        """Check number of nodes initialized."""
        num_nodes = len(self.graph)
        num_tumor = len({name for kind, name in self.graph if kind == "tumor"})
        num_lnls = len({name for kind, name in self.graph if kind == "lnl"})

        self.assertEqual(len(self.model.nodes), num_nodes)
        self.assertEqual(len(self.model.tumors), num_tumor)
        self.assertEqual(len(self.model.lnls), num_lnls)

    def test_num_edges(self):
        """Check number of edges initialized."""
        num_edges = sum(len(receiving_nodes) for receiving_nodes in self.graph.values())
        num_tumor_edges = sum(
            len(receiving_nodes) for (kind, _), receiving_nodes in self.graph.items()
            if kind == "tumor"
        )
        num_lnl_edges = sum(
            len(receiving_nodes) for (kind, _), receiving_nodes in self.graph.items()
            if kind == "lnl"
        )

        self.assertEqual(len(self.model.edges), num_edges)
        self.assertEqual(len(self.model.tumor_edges), num_tumor_edges)
        self.assertEqual(len(self.model.lnl_edges), num_lnl_edges)
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
            self.assertEqual(edge.parent.name, "T")
            self.assertIn(edge.child.name, receiving_lnls)
            self.assertTrue(edge.is_tumor_spread)
            self.assertIn(edge.name, connecting_edge_names)


class ParameterAssignmentTestCase(FixtureMixin, unittest.TestCase):
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
        self.model.assign_params(**params_to_set)
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
        first_lnl_name = self.model.lnls[0].name
        _ = self.model.transition_matrix
        self.assertTrue(hasattr(self.model, "_transition_matrix"))
        self.model.edge_params[f"spread_T_to_{first_lnl_name}"].set(0.5)
        self.assertFalse(hasattr(self.model, "_transition_matrix"))


class TransitionMatrixTestCase(FixtureMixin, unittest.TestCase):
    """Test the generation of the transition matrix in a binary model."""

    def setUp(self):
        """Initialize a simple binary model."""
        super().setUp()

        params_to_set = self.create_random_params(seed=42)
        self.model.assign_params(**params_to_set)

    def test_shape(self):
        """Make sure the transition matrix has the correct shape."""
        num_lnls = len({name for kind, name in self.graph if kind == "lnl"})
        self.assertEqual(self.model.transition_matrix.shape, (2**num_lnls, 2**num_lnls))

    def test_is_probabilistic(self):
        """Make sure the rows of the transition matrix sum to one."""
        row_sums = np.sum(self.model.transition_matrix, axis=1)
        self.assertTrue(np.allclose(row_sums, 1.))

    @staticmethod
    def is_recusively_upper_triangular(mat: np.ndarray) -> bool:
        """Return `True` is `mat` is recursively upper triangular."""
        if mat.shape == (1, 1):
            return True

        if not np.all(np.equal(np.triu(mat), mat)):
            return False

        half = mat.shape[0] // 2
        for i in [0, 1]:
            for j in [0, 1]:
                return TransitionMatrixTestCase.is_recusively_upper_triangular(
                    mat[i * half:(i + 1) * half, j * half:(j + 1) * half]
                )

    def test_is_recusively_upper_triangular(self) -> None:
        """Make sure the transition matrix is recursively upper triangular."""
        self.assertTrue(self.is_recusively_upper_triangular(self.model.transition_matrix))


class ObservationMatrixTestCase(FixtureMixin, unittest.TestCase):
    """Test the generation of the observation matrix in a binary model."""

    def setUp(self):
        super().setUp()

        ct_sp, ct_sn = 0.81, 0.86
        mr_sp, mr_sn = 0.85, 0.82

        self.model.modalities = {
            "CT": np.array([
                [ct_sp     , 1. - ct_sp],
                [1. - ct_sn, ct_sn     ],
            ]),
            "MR": np.array([
                [mr_sp     , 1. - mr_sp],
                [1. - mr_sn, mr_sn     ],
            ]),
        }

    def test_shape(self):
        """Make sure the observation matrix has the correct shape."""
        num_lnls = len(self.model.lnls)
        self.assertEqual(self.model.observation_matrix.shape, (2**num_lnls, 2**num_lnls))



if __name__ == "__main__":
    fixture = FixtureMixin()
    fixture.setUp()

    params = fixture.create_random_params(234)
    fixture.model.assign_params(**params)
    _ = fixture.model.transition_matrix
    del fixture.model.transition_matrix
    _ = fixture.model.transition_matrix

