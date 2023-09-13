"""
Test the bilateral model.
"""
import unittest

import fixtures
import numpy as np

from lymph.descriptors.modalities import Clinical


class BilateralInitTest(fixtures.BilateralModelMixin, unittest.TestCase):
    """Test the delegation of attrs from the unilateral class to the bilateral one."""

    def setUp(self):
        super().setUp()
        self.load_patient_data()

    def test_delegation(self):
        """Test that the unilateral model delegates the attributes."""
        self.assertEqual(
            self.model.is_binary, self.model.ipsi.is_binary
        )
        self.assertEqual(
            self.model.is_trinary, self.model.ipsi.is_trinary
        )
        self.assertEqual(
            self.model.max_time, self.model.ipsi.max_time
        )
        self.assertEqual(
            list(self.model.t_stages), list(self.model.ipsi.t_stages)
        )

    def test_lnl_edge_sync(self):
        """Check if synced LNL edges update their respective parameters."""
        rng = np.random.default_rng(42)
        for ipsi_edge in self.model.ipsi.graph.lnl_edges.values():
            contra_edge = self.model.contra.graph.lnl_edges[ipsi_edge.name]
            ipsi_edge.set_params(spread=rng.random())
            self.assertEqual(
                ipsi_edge.get_params("spread"),
                contra_edge.get_params("spread"),
            )

    def test_modality_sync(self):
        """Make sure the modalities are synced between the two sides."""
        rng = np.random.default_rng(42)
        self.model.ipsi.modalities = {"foo": Clinical(
            specificity=rng.uniform(),
            sensitivity=rng.uniform(),
        )}
        self.assertEqual(
            self.model.ipsi.modalities["foo"].sensitivity,
            self.model.contra.modalities["foo"].sensitivity,
        )
        self.assertEqual(
            self.model.ipsi.modalities["foo"].specificity,
            self.model.contra.modalities["foo"].specificity,
        )


class ParameterAssignmentTestCase(fixtures.BilateralModelMixin, unittest.TestCase):
    """Test the parameter assignment."""

    def setUp(self):
        res = super().setUp()
        self.create_random_params(42)
        return res
