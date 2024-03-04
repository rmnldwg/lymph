"""
Test the midline model for the binary case.
"""
import unittest
from typing import Literal

import numpy as np
import pandas as pd

from lymph import models

from . import fixtures


class MidlineSetParamsTestCase(unittest.TestCase):
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

        self.model = models.Midline(
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
        self.assertFalse(self.model.is_trinary)


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


class MidlineLikelihoodTestCase(unittest.TestCase):
    """Check that the likelihood function works correctly."""

    def setUp(
        self,
        seed: int = 42,
        graph_size: Literal["small", "medium", "large"] = "small",
        use_mixing: bool = True,
        use_central: bool = False,
        use_midext_evo: bool = True,
        is_symmetric: dict[str, bool] | None = None,
    ) -> None:
        super().setUp()
        self.rng = np.random.default_rng(seed)
        graph_dict = fixtures.get_graph(graph_size)
        if is_symmetric is None:
            is_symmetric = {"tumor_spread": False, "lnl_spread": True}

        self.model = models.Midline(
            graph_dict=graph_dict,
            is_symmetric=is_symmetric,
            use_mixing=use_mixing,
            use_central=use_central,
            use_midext_evo=use_midext_evo,
        )
        self.model.set_distribution(
            "early",
            fixtures.create_random_dist(
                type_="frozen",
                max_time=self.model.max_time,
                rng=self.rng,
            ),
        )
        self.model.set_distribution(
            "late",
            fixtures.create_random_dist(
                type_="parametric",
                max_time=self.model.max_time,
                rng=self.rng,
            ),
        )
        self.model.set_modality("pathology", spec=1., sens=1., kind="pathological")
        self.model.load_patient_data(pd.read_csv("./tests/data/2021-clb-oropharynx.csv", header=[0,1,2]))


    def test_likelihood(self) -> None:
        """Check that the likelihood function works correctly."""
        params_to_set = {k: self.rng.uniform() for k in self.model.get_params().keys()}
        self.model.set_params(**params_to_set)

        # Check that the likelihood is a number
        self.assertTrue(np.isscalar(self.model.likelihood()))

        # Check that the likelihood is not NaN
        self.assertFalse(np.isnan(self.model.likelihood()))

        # Check that the log-likelihood is smaller than 0
        self.assertLessEqual(self.model.likelihood(), 0)


class MidlineDrawPatientsTestCase(unittest.TestCase):
    """Check the data generation."""

    def setUp(self) -> None:
        super().setUp()
        self.rng = np.random.default_rng(42)
        graph_dict = {
            ("tumor", "T"): ["A"],
            ("lnl", "A"): ["B"],
            ("lnl", "B"): [],
        }
        self.model = models.Midline(
            graph_dict=graph_dict,
            use_mixing=True,
            use_central=False,
            use_midext_evo=True,
            marginalize_unknown=False,
            unilateral_kwargs={"max_time": 2},
        )
        self.model.set_distribution("early", [0., 1., 0.])
        self.model.set_distribution("late", [0., 0., 1.])
        self.model.set_modality("pathology", spec=1., sens=1., kind="pathological")


    def test_draw_patients(self) -> None:
        """Check that the data generation works correctly."""
        self.model.set_params(
            ipsi_TtoA_spread=1.0,
            contra_TtoA_spread=0.0,
            AtoB_spread=1.0,
            mixing=0.5,
            midext_prob=0.5,
        )
        drawn_data = self.model.draw_patients(
            num=100,
            stage_dist=[0.5, 0.5],
            rng=self.rng,
        )
        self.assertEqual(len(drawn_data), 100)
