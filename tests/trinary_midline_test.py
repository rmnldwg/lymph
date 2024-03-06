"""
Test the midline model for the binary case.
"""
from typing import Literal

import numpy as np

from lymph import models

from . import fixtures


class MidlineSetParamsTestCase(fixtures.IgnoreWarningsTestCase):
    """Check that the complex parameter assignment works correctly."""

    def setUp(
        self,
        seed: int = 42,
        graph_size: Literal["small", "medium", "large"] = "small",
        use_mixing: bool = True,
        use_central: bool = True,
        is_symmetric: dict[str, bool] | None = None,
    ) -> None:
        super().setUp()
        self.rng = np.random.default_rng(seed)
        graph_dict = fixtures.get_graph(graph_size)
        if is_symmetric is None:
            is_symmetric = {"tumor_spread": False, "lnl_spread": True}

        self.model = models.Midline.trinary(
            graph_dict=graph_dict,
            is_symmetric=is_symmetric,
            use_mixing=use_mixing,
            use_central=use_central,
            use_midext_evo=False,
        )


    def test_init(self) -> None:
        """Check some basic attributes."""
        self.assertTrue(self.model.use_central)
        self.assertTrue(self.model.use_mixing)
        self.assertTrue(self.model.is_trinary)


    def test_set_spread_params(self) -> None:
        """Check that the complex parameter assignment works correctly."""
        params_to_set = {k: self.rng.uniform() for k in self.model.get_params().keys()}
        self.model.set_params(**params_to_set)

        self.assertEqual(
            self.model.central.ipsi.get_tumor_spread_params(),
            self.model.central.contra.get_tumor_spread_params(),
        )
        self.assertEqual(
            self.model.central.ipsi.get_lnl_spread_params(),
            self.model.central.contra.get_lnl_spread_params(),
        )
        self.assertEqual(
            self.model.central.contra.get_lnl_spread_params(),
            self.model.ext.ipsi.get_lnl_spread_params(),
        )
        self.assertEqual(
            self.model.ext.ipsi.get_lnl_spread_params(),
            self.model.noext.ipsi.get_lnl_spread_params(),
        )
        self.assertEqual(
            self.model.ext.ipsi.get_tumor_spread_params(),
            self.model.noext.ipsi.get_tumor_spread_params(),
        )
