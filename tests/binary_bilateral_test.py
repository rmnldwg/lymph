"""
Test the bilateral model.
"""
import unittest

import numpy as np
from binary_unilateral_test import LoadDataFixtureMixin

import lymph
from lymph.descriptors.modalities import Clinical


class BilateralModelFixtureMixin(LoadDataFixtureMixin):
    """Mixin for testing the bilateral model."""

    def setUp(self):
        super().setUp()
        self.bi_model = lymph.models.Bilateral(self.graph_dict)
        self.bi_model.ipsi.graph.edges["II_to_III"].set_params(spread=0.123)

    def create_random_params(self, seed: int = 42) -> dict[str, float]:
        """Create a random set of parameters."""
        rng = np.random.default_rng(seed)
        self.bi_model.diag_time_dists["early"] = self.create_frozen_diag_time_dist(rng)
        self.bi_model.diag_time_dists["late"] = self.create_parametric_diag_time_dist(rng)
        ipsi_params = super().create_random_params(seed=rng)
        contra_params = super().create_random_params(seed=rng)
        return None


class BilateralInitTest(BilateralModelFixtureMixin, unittest.TestCase):
    """Test the delegation of attrs from the unilateral class to the bilateral one."""

    def test_delegation(self):
        """Test that the unilateral model delegates the attributes."""
        self.assertEqual(
            self.bi_model.is_binary, self.bi_model.ipsi.is_binary
        )
        self.assertEqual(
            self.bi_model.is_trinary, self.bi_model.ipsi.is_trinary
        )
        self.assertEqual(
            self.bi_model.max_time, self.bi_model.ipsi.max_time
        )
        self.assertEqual(
            list(self.bi_model.t_stages), list(self.bi_model.ipsi.t_stages)
        )

    def test_lnl_edge_sync(self):
        """Check if synced LNL edges update their respective parameters."""
        rng = np.random.default_rng(42)
        for ipsi_edge in self.bi_model.ipsi.graph.lnl_edges.values():
            contra_edge = self.bi_model.contra.graph.lnl_edges[ipsi_edge.name]
            ipsi_edge.set_params(spread=rng.random())
            self.assertEqual(
                ipsi_edge.get_params("spread"),
                contra_edge.get_params("spread"),
            )

    def test_modality_sync(self):
        """Make sure the modalities are synced between the two sides."""
        rng = np.random.default_rng(42)
        self.bi_model.ipsi.modalities = {"foo": Clinical(
            specificity=rng.uniform(),
            sensitivity=rng.uniform(),
        )}
        self.assertEqual(
            self.bi_model.ipsi.modalities["foo"].sensitivity,
            self.bi_model.contra.modalities["foo"].sensitivity,
        )
        self.assertEqual(
            self.bi_model.ipsi.modalities["foo"].specificity,
            self.bi_model.contra.modalities["foo"].specificity,
        )


class ParameterAssignmentTestCase(BilateralModelFixtureMixin, unittest.TestCase):
    """Test the parameter assignment."""

    def setUp(self):
        res = super().setUp()
        self.create_random_params(42)
        return res
