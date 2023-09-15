"""
Test the bilateral model.
"""
import unittest

import fixtures
import numpy as np

from lymph.modalities import Clinical


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


class ModalityDelegationTestCase(fixtures.BilateralModelMixin, unittest.TestCase):
    """Make sure the modality is delegated from the ipsi side correctly."""

    def setUp(self):
        super().setUp()
        self.model.modalities = fixtures.MODALITIES

    def test_modality_access(self):
        """Test that the modality can be accessed."""
        self.assertEqual(
            self.model.modalities["CT"].sensitivity,
            self.model.ipsi.modalities["CT"].sensitivity,
        )
        self.assertEqual(
            self.model.modalities["FNA"].specificity,
            self.model.ipsi.modalities["FNA"].specificity,
        )

    def test_modality_delete(self):
        """Test that the modality can be deleted."""
        del self.model.modalities["CT"]
        self.assertNotIn("CT", self.model.modalities)
        self.assertNotIn("CT", self.model.ipsi.modalities)
        self.assertNotIn("CT", self.model.contra.modalities)

    def test_modality_update(self):
        """Test that the modality can be updated."""
        self.model.modalities["CT"].sensitivity = 0.8
        self.assertEqual(
            self.model.modalities["CT"].sensitivity,
            self.model.ipsi.modalities["CT"].sensitivity,
        )
        self.assertEqual(
            self.model.modalities["CT"].sensitivity,
            self.model.contra.modalities["CT"].sensitivity,
        )

    def test_modality_reset(self):
        """Test resetting the modalities also works."""
        self.model.modalities = {"foo": Clinical(0.8, 0.9)}
        self.assertEqual(
            self.model.modalities["foo"].sensitivity,
            self.model.ipsi.modalities["foo"].sensitivity,
        )
        self.assertEqual(
            self.model.modalities["foo"].specificity,
            self.model.contra.modalities["foo"].specificity,
        )


class LikelihoodTestCase(fixtures.BilateralModelMixin, unittest.TestCase):
    """Check that the (log-)likelihood is computed correctly."""

    def setUp(self):
        super().setUp()
        self.model.modalities = fixtures.MODALITIES
        self.load_patient_data()

    def test_compute_likelihood_twice(self):
        """Test that the likelihood is computed correctly."""
        first_llh = self.model.likelihood(log=True)
        second_llh = self.model.likelihood(log=True)
        self.assertEqual(first_llh, second_llh)
