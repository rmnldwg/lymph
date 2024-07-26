"""Unit tests for the Node classes."""
import unittest

from lymph import graph


class BinaryLymphNodeLevelTestCase(unittest.TestCase):
    """Test case for a binary lymph node level."""

    def setUp(self) -> None:
        super().setUp()
        self.initial_name = "test"
        self.initial_state = 0
        self.lnl = graph.LymphNodeLevel.binary(
            self.initial_name,
            state=self.initial_state,
        )

    def test_binary_init(self) -> None:
        """Test the binary node initialization."""
        self.assertEqual(self.lnl.name, self.initial_name)
        self.assertEqual(self.lnl.state, self.initial_state)
        self.assertEqual(self.lnl.allowed_states, [0, 1])
        self.assertTrue(self.lnl.is_binary)

    def test_str(self) -> None:
        """Test the string representation of the node."""
        self.assertEqual(str(self.lnl), f"binary LNL '{self.initial_name}'")

    def test_repr(self) -> None:
        """Test if the node can be recreated from repr."""
        # pylint: disable=eval-used
        recreated_lnl = eval(repr(self.lnl), {"LymphNodeLevel": graph.LymphNodeLevel})
        self.assertEqual(self.lnl.name, recreated_lnl.name)
        self.assertEqual(self.lnl.state, recreated_lnl.state)
        self.assertEqual(self.lnl.allowed_states, recreated_lnl.allowed_states)
