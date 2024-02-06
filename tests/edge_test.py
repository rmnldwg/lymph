"""Unit tests for the Edge class."""
import unittest

import numpy as np

from lymph import graph


class BinaryEdgeTestCase(unittest.TestCase):
    """Tests for the Edge class."""

    def setUp(self) -> None:
        super().setUp()
        parent = graph.LymphNodeLevel("parent")
        child = graph.LymphNodeLevel("child")
        self.was_called = False
        self.edge = graph.Edge(parent, child, callbacks=[self.callback])

    def callback(self) -> None:
        """Callback function for the edge."""
        self.was_called = True

    def test_str(self) -> None:
        """Test the string representation of the edge."""
        self.assertEqual(str(self.edge), "Edge parent to child")

    def test_repr(self) -> None:
        """Test if the edge can be recreated from repr."""
        # pylint: disable=eval-used
        recreated_edge = eval(
            repr(self.edge),
            {
                "Edge": graph.Edge,
                "LymphNodeLevel": graph.LymphNodeLevel,
                "Tumor": graph.Tumor,
            },
        )
        self.assertEqual(self.edge.name, recreated_edge.name)
        self.assertEqual(self.edge.parent.name, recreated_edge.parent.name)
        self.assertEqual(self.edge.child.name, recreated_edge.child.name)
        self.assertEqual(self.edge.spread_prob, recreated_edge.spread_prob)
        self.assertEqual(self.edge.micro_mod, recreated_edge.micro_mod)

    def test_callback_on_param_change(self) -> None:
        """Test if the callback function is called."""
        self.edge.spread_prob = 0.5
        self.assertTrue(self.was_called)

    def test_graph_change(self) -> None:
        """Check if the callback also works when parent/child nodes are changed."""
        old_child = self.edge.child
        new_child = graph.LymphNodeLevel("new_child")
        self.edge.child = new_child
        self.assertTrue(self.was_called)
        self.assertNotIn(self.edge, old_child.inc)

    def test_transition_tensor_row_sums(self) -> None:
        """Testing the transition tensor."""
        row_sum = self.edge.transition_tensor.sum(axis=2)
        self.assertTrue(np.allclose(row_sum, 1.0))


class TrinaryEdgeTestCase(unittest.TestCase):
    """Tests for the Edge class in case the parent and child node are trinary."""

    def setUp(self) -> None:
        super().setUp()
        parent = graph.LymphNodeLevel("parent", allowed_states=[0,1,2])
        child = graph.LymphNodeLevel("child", allowed_states=[0,1,2])
        self.edge = graph.Edge(parent, child)
        self.edge.spread_prob = 0.3
        self.edge.micro_mod = 0.7

        self.growth_edge = graph.Edge(parent, parent)
        self.growth_edge.spread_prob = 0.5

    def test_transition_tensor_row_sums(self) -> None:
        """Testing the transition tensor."""
        row_sum = self.edge.transition_tensor.sum(axis=2)
        self.assertTrue(np.allclose(row_sum, 1.0))

    def test_growth_transition_tensor_row_sums(self) -> None:
        """Testing the transition tensor."""
        row_sum = self.growth_edge.transition_tensor.sum(axis=2)
        self.assertTrue(np.allclose(row_sum, 1.0))
