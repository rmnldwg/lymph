"""Test the unilateral module."""
import unittest

import numpy as np

from lymph import Unilateral


class TestUnilateral(unittest.TestCase):
    """Test the unilateral module."""

    def setUp(self) -> None:
        """Set up the test case."""
        seed = 42
        rng = np.random.default_rng(seed=seed)

        graph = {
            ("tumor", "T"): ["I", "II", "III", "IV"],
            ("lnl", "I"): [],
            ("lnl", "II"): ["I", "III"],
            ("lnl", "III"): ["IV"],
            ("lnl", "IV"): [],
        }
        self.binary_unilateral = Unilateral(graph=graph)
        self.trinary_unilateral = Unilateral(graph=graph, allowed_states=[0, 1, 2])

        self.params_dict = {
            "spread_T_to_I": rng.uniform(),
            "spread_T_to_II": rng.uniform(),
            "spread_T_to_III": rng.uniform(),
            "spread_T_to_IV": rng.uniform(),
            "spread_II_to_I": rng.uniform(),
            "spread_II_to_III": rng.uniform(),
            "spread_III_to_IV": rng.uniform(),
        }

    def test_init_binary(self) -> None:
        """Test the initialization of a binary Unilateral class."""
        self.assertEqual(len(self.binary_unilateral.tumors), 1)
        self.assertEqual(len(self.binary_unilateral.lnls), 4)
        self.assertEqual(len(self.binary_unilateral.tumor_edges), 4)
        self.assertEqual(len(self.binary_unilateral.lnl_edges), 3)
        self.assertTrue(self.binary_unilateral.is_binary)

    def test_init_trinary(self) -> None:
        """Test the initialization of a binary Unilateral class."""
        self.assertTrue(self.trinary_unilateral.is_trinary)

    def test_get_and_set_params(self) -> None:
        """Test the get and set params methods."""
        self.binary_unilateral.assign_parameters(**self.params_dict)
        for expected_value, set_value in zip(
            self.params_dict.values(),
            self.binary_unilateral.get_parameters().values()
        ):
            self.assertEqual(expected_value, set_value)


if __name__ == "__main__":
    unittest.main()
