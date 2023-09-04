"""
Test the bilateral model.
"""
import unittest

from binary_unilateral_test import ModelFixtureMixin

import lymph


class BilateralInitTest(ModelFixtureMixin, unittest.TestCase):
    """Test the delegation of attrs from the unilateral class to the bilateral one."""

    def setUp(self):
        super().setUp()
        self.bilateral_model = lymph.models.Bilateral(self.graph_dict)

    def test_delegation(self):
        """Test that the unilateral model delegates the attributes."""
        self.assertEqual(
            self.bilateral_model.is_binary, self.bilateral_model.ipsi.is_binary
        )
        self.assertEqual(
            self.bilateral_model.is_trinary, self.bilateral_model.ipsi.is_trinary
        )
        self.assertEqual(
            self.bilateral_model.max_time, self.bilateral_model.ipsi.max_time
        )
        self.assertEqual(
            list(self.bilateral_model.t_stages), list(self.bilateral_model.ipsi.t_stages)
        )
