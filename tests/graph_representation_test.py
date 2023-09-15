"""Test the graph representation class of the package."""
import unittest

import numpy as np

from lymph import graph


class ConstructBinaryGraphRepresentationTestCase(unittest.TestCase):
    """Test suite for the graph representation class."""

    def setUp(self) -> None:
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
        self.graph_dict = large_graph
        self.graph_repr = graph.Representation(
            graph_dict=self.graph_dict,
            allowed_states=[0, 1],
            on_edge_change=[self.callback],
        )
        self.was_called = False
        self.rng = np.random.default_rng(42)

    def callback(self) -> None:
        """Callback function for the graph."""
        self.was_called = True

    def test_nodes(self) -> None:
        """Test the number of nodes."""
        self.assertEqual(len(self.graph_repr.nodes), len(self.graph_dict))
        node_names = {tpl[1] for tpl in self.graph_dict.keys()}

        for name, node in self.graph_repr.nodes.items():
            self.assertIn(name, node_names)
            if isinstance(node, graph.Tumor):
                self.assertFalse(hasattr(node, "inc"))
                self.assertTrue(hasattr(node, "out"))
            else:
                self.assertTrue(hasattr(node, "inc"))

    def test_edges(self) -> None:
        """Test the number of edges."""
        num_edges = 0
        for _, children in self.graph_dict.items():
            num_edges += len(set(children))
        self.assertEqual(len(self.graph_repr.edges), num_edges)

        for edge in self.graph_repr.edges.values():
            self.assertTrue(isinstance(edge.parent, graph.AbstractNode))
            self.assertTrue(isinstance(edge.child, graph.AbstractNode))
