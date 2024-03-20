"""
Test the midline model for the binary case.
"""

import numpy as np
import pandas as pd

from lymph import models

from . import fixtures


class MidlineSetParamsTestCase(
    fixtures.MidlineFixtureMixin,
    fixtures.IgnoreWarningsTestCase,
):
    """Check that the complex parameter assignment works correctly."""
    def setUp(self):
        return super().setUp(use_central=True, use_midext_evo=False)

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


    def test_get_set_params_order(self) -> None:
        """Check if the order of getter and setter is the same."""
        num_dims = self.model.get_num_dims()
        params_to_set = np.linspace(0., 1., num_dims + 1)
        unused_param = self.model.set_params(*params_to_set)
        returned_params = list(self.model.get_params(as_dict=False))

        self.assertEqual(unused_param, params_to_set[-1])
        self.assertEqual(params_to_set[:-1].tolist(), returned_params)


class MidlineLikelihoodTestCase(
    fixtures.MidlineFixtureMixin,
    fixtures.IgnoreWarningsTestCase,
):
    """Check that the likelihood function works correctly."""

    def setUp(self) -> None:
        """Set up the test case."""
        super().setUp()
        self.init_diag_time_dists(early="frozen", late="parametric")
        self.model.set_modality("pathology", spec=1., sens=1., kind="pathological")
        self.model.load_patient_data(
            pd.read_csv("./tests/data/2021-clb-oropharynx.csv", header=[0,1,2]),
        )


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


class MidlineRiskTestCase(
    fixtures.MidlineFixtureMixin,
    fixtures.IgnoreWarningsTestCase,
):
    """Check that the risk method works correctly."""

    def setUp(self) -> None:
        """Set up the test case."""
        super().setUp()
        self.init_diag_time_dists(early="frozen", late="parametric")
        self.model.set_modality("pathology", spec=1., sens=1., kind="pathological")
        self.model.set_params(
            midext_prob=0.1,
            ipsi_TtoII_spread=0.35,
            ipsi_TtoIII_spread=0.0,
            contra_TtoII_spread=0.05,
            contra_TtoIII_spread=0.0,
            mixing=0.5,
            IItoIII_spread=0.1,
            late_p=0.5,
        )


    def test_risk(self) -> None:
        """Check that the risk method works correctly."""
        plain_risk = self.model.risk()
        self.assertEqual(plain_risk.shape, (4,4))
        self.assertTrue(np.isclose(plain_risk.sum(), 1.0))
        self.assertTrue(np.allclose(plain_risk[1,:], 0.))
        self.assertTrue(np.allclose(plain_risk[:,1], 0.))

        lnlIII_risk = self.model.risk(involvement={"ipsi": {"II": False, "III": True}})
        self.assertTrue(np.isscalar(lnlIII_risk))
        self.assertAlmostEqual(lnlIII_risk, 0.0)

        ipsi_lnlII_risk = self.model.risk(involvement={"ipsi": {"II": True}})
        contra_lnlII_risk = self.model.risk(involvement={"contra": {"II": True}})
        self.assertGreater(ipsi_lnlII_risk, contra_lnlII_risk)
        ext_contra_lnlII_risk = self.model.risk(
            involvement={"contra": {"II": True}},
            midext=True,
        )
        self.assertGreater(ext_contra_lnlII_risk, contra_lnlII_risk)
        noext_contra_lnlII_risk = self.model.risk(
            involvement={"contra": {"II": True}},
            midext=False,
        )
        self.assertGreater(contra_lnlII_risk, noext_contra_lnlII_risk)
        self.assertGreater(ext_contra_lnlII_risk, noext_contra_lnlII_risk)


    def test_risk_given_state_dist(self) -> None:
        """Check how providing a state distribution works correctly."""
        state_dist_3d = self.model.state_dist(t_stage="early")
        self.assertEqual(state_dist_3d.shape, (2, 4, 4))

        risk_from_state_dist = self.model.risk(given_state_dist=state_dist_3d, midext=True)
        risk_direct = self.model.risk(midext=True)
        self.assertTrue(np.allclose(risk_from_state_dist, risk_direct))

        state_dist_2d = state_dist_3d[0] / state_dist_3d[0].sum()
        risk_from_state_dist = self.model.risk(given_state_dist=state_dist_2d)
        risk_direct = self.model.risk(midext=False)
        self.assertTrue(np.allclose(risk_from_state_dist, risk_direct))

        state_dist_2d = state_dist_3d[1] / state_dist_3d[1].sum()
        risk_from_state_dist = self.model.risk(given_state_dist=state_dist_2d)
        risk_direct = self.model.risk(midext=True)
        self.assertTrue(np.allclose(risk_from_state_dist, risk_direct))



class MidlineDrawPatientsTestCase(fixtures.IgnoreWarningsTestCase):
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
            uni_kwargs={"max_time": 2},
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
