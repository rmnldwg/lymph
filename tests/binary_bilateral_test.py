"""
Test the bilateral model.
"""

import numpy as np

from lymph import models
from lymph.utils import flatten

from . import fixtures


class BilateralInitTest(fixtures.BilateralModelMixin, fixtures.IgnoreWarningsTestCase):
    """Test the delegation of attrs from the unilateral class to the bilateral one."""

    def setUp(self):
        self.model_kwargs = {"is_symmetric": {
            "tumor_spread": True,
            "lnl_spread": True,
        }}
        super().setUp()
        self.load_patient_data()

    def test_delegation(self):
        """Test that the unilateral model delegates the attributes."""
        self.assertEqual(self.model.is_binary, self.model.ipsi.is_binary)
        self.assertEqual(self.model.is_trinary, self.model.ipsi.is_trinary)
        self.assertEqual(self.model.max_time, self.model.ipsi.max_time)
        self.assertEqual(list(self.model.t_stages), list(self.model.ipsi.t_stages))

    def test_transition_matrix_sync(self):
        """Make sure contra transition matrix gets recomputed when ipsi param is set."""
        ipsi_trans_mat = self.model.ipsi.transition_matrix()
        contra_trans_mat = self.model.contra.transition_matrix()
        rand_ipsi_param = self.rng.choice(list(
            self.model.ipsi.get_params(as_dict=True).keys()
        ))
        self.model.set_params(**{f"ipsi_{rand_ipsi_param}": self.rng.random()})
        self.assertFalse(np.all(
            ipsi_trans_mat == self.model.ipsi.transition_matrix()
        ))
        self.assertFalse(np.all(
            contra_trans_mat == self.model.contra.transition_matrix()
        ))

    def test_modality_sync(self):
        """Make sure the modalities are synced between the two sides."""
        self.model.set_modality("foo", spec=self.rng.uniform(), sens=self.rng.uniform())
        self.assertEqual(
            self.model.ipsi.get_modality("foo").sens,
            self.model.contra.get_modality("foo").sens,
        )
        self.assertEqual(
            self.model.ipsi.get_modality("foo").spec,
            self.model.contra.get_modality("foo").spec,
        )

    def test_asymmetric_model(self):
        """Check if different graphs work for the ipsi and contra side."""
        ipsi_graph = fixtures.get_graph("medium")
        contra_graph = fixtures.get_graph("small")

        model = models.Bilateral(
            graph_dict=ipsi_graph,
            contra_kwargs={"graph_dict": contra_graph},
        )

        self.assertEqual(
            list(model.ipsi.graph.nodes.keys()),
            [key[1] for key in ipsi_graph.keys()],
        )
        self.assertEqual(
            list(model.contra.graph.nodes.keys()),
            [key[1] for key in contra_graph.keys()],
        )
        self.assertEqual(
            len(model.ipsi.get_params()),
            sum(len(val) for val in ipsi_graph.values()),
        )
        self.assertEqual(
            len(model.contra.get_params()),
            sum(len(val) for val in contra_graph.values()),
        )


class ModalityDelegationTestCase(
    fixtures.BilateralModelMixin,
    fixtures.IgnoreWarningsTestCase,
):
    """Make sure the modality is delegated from the ipsi side correctly."""

    def setUp(self):
        super().setUp()
        self.model.replace_all_modalities(fixtures.MODALITIES)

    def test_modality_access(self):
        """Test that the modality can be accessed."""
        self.assertEqual(
            self.model.get_modality("CT").sens,
            self.model.ipsi.get_modality("CT").sens,
        )
        self.assertEqual(
            self.model.get_modality("FNA").spec,
            self.model.ipsi.get_modality("FNA").spec,
        )

    def test_modality_delete(self):
        """Test that the modality can be deleted."""
        self.model.del_modality("CT")
        self.assertNotIn("CT", self.model.get_all_modalities())
        self.assertNotIn("CT", self.model.ipsi.get_all_modalities())
        self.assertNotIn("CT", self.model.contra.get_all_modalities())

    def test_modality_update(self):
        """Test that the modality can be updated."""
        old_mod = self.model.get_modality("CT")
        self.model.set_modality("CT", spec=old_mod.spec, sens=0.8)
        self.assertEqual(
            self.model.get_modality("CT").sens,
            self.model.ipsi.get_modality("CT").sens,
        )
        self.assertEqual(
            self.model.get_modality("CT").sens,
            self.model.contra.get_modality("CT").sens,
        )

    def test_modality_reset(self):
        """Test resetting the modalities also works."""
        self.model.set_modality("foo", spec=0.8, sens=0.9)
        self.assertEqual(
            self.model.get_modality("foo").sens,
            self.model.ipsi.get_modality("foo").sens,
        )
        self.assertEqual(
            self.model.get_modality("foo").spec,
            self.model.contra.get_modality("foo").spec,
        )

    def test_diag_time_dists_delegation(self):
        """Test that the diagnose time distributions are delegated."""
        self.assertEqual(
            list(self.model.get_distribution("early").pmf),
            list(self.model.ipsi.get_distribution("early").pmf),
        )
        self.assertEqual(
            list(self.model.get_distribution("late").get_params()),
            list(self.model.ipsi.get_distribution("late").get_params()),
        )
        self.assertEqual(
            list(self.model.get_distribution("early").pmf),
            list(self.model.contra.get_distribution("early").pmf),
        )
        self.assertEqual(
            list(self.model.get_distribution("late").get_params()),
            list(self.model.contra.get_distribution("late").get_params()),
        )


class NoSymmetryParamsTestCase(
    fixtures.BilateralModelMixin,
    fixtures.IgnoreWarningsTestCase,
):
    """Test the parameter assignment when the model is not symmetric"""

    def setUp(self):
        self.model_kwargs = {
            "is_symmetric": {
                "tumor_spread": False,
                "lnl_spread": False,
            }
        }
        super().setUp()

    def test_get_params_as_args(self):
        """Test that the parameters can be retrieved."""
        ipsi_args = list(self.model.ipsi.get_params(as_dict=False))
        contra_args = list(self.model.contra.get_params(as_dict=False))
        both_args = list(self.model.get_params(as_dict=False))
        num_param_dists = len(self.model.get_distribution_params())
        # need plus one, because distribution's parameter is accounted for twice
        self.assertEqual(len(ipsi_args) + len(contra_args), len(both_args) + 1)
        self.assertEqual(
            [*ipsi_args[:-num_param_dists], *contra_args[:-num_param_dists]],
            both_args[:-num_param_dists],
        )

    def test_get_params_as_dict(self):
        """Test that the parameters can be retrieved."""
        ipsi_dict = self.model.ipsi.get_params(as_dict=True)
        contra_dict = self.model.contra.get_params(as_dict=True)
        both_dict = self.model.get_params(as_dict=True, as_flat=False)
        dist_param_keys = self.model.get_distribution_params().keys()

        for key in dist_param_keys:
            ipsi_dict.pop(key)
            contra_dict.pop(key)

        self.assertEqual(ipsi_dict, flatten(both_dict["ipsi"]))
        self.assertEqual(contra_dict, flatten(both_dict["contra"]))

    def test_set_params_as_args(self):
        """Test that the parameters can be set."""
        ipsi_tumor_spread_args = self.rng.uniform(size=len(self.model.ipsi.graph.tumor_edges))
        ipsi_lnl_spread_args = self.rng.uniform(size=len(self.model.ipsi.graph.lnl_edges))
        contra_tumor_spread_args = self.rng.uniform(size=len(self.model.contra.graph.tumor_edges))
        contra_lnl_spread_args = self.rng.uniform(size=len(self.model.contra.graph.lnl_edges))
        dist_params = self.rng.uniform(size=len(self.model.get_distribution_params()))

        self.model.set_params(
            *ipsi_tumor_spread_args,
            *contra_tumor_spread_args,
            *ipsi_lnl_spread_args,
            *contra_lnl_spread_args,
            *dist_params,
        )
        self.assertEqual(
            [*ipsi_tumor_spread_args, *ipsi_lnl_spread_args, *dist_params],
            list(self.model.ipsi.get_params(as_dict=False)),
        )
        self.assertEqual(
            [*contra_tumor_spread_args, *contra_lnl_spread_args, *dist_params],
            list(self.model.contra.get_params(as_dict=False)),
        )
        self.assertEqual(
            list(self.model.ipsi.get_distribution("late").get_params())[0],
            list(self.model.contra.get_distribution("late").get_params())[0],
        )

    def test_set_params_as_dict(self):
        """Test that the parameters can be set via keyword arguments."""
        params_to_set = {k: self.rng.uniform() for k in self.model.get_params().keys()}
        self.model.set_params(**params_to_set)
        self.assertEqual(params_to_set, self.model.get_params())


class SymmetryParamsTestCase(
    fixtures.BilateralModelMixin,
    fixtures.IgnoreWarningsTestCase,
):
    """Test the parameter assignment when the model is symmetric."""

    def setUp(self):
        self.model_kwargs = {
            "is_symmetric": {
                "tumor_spread": True,
                "lnl_spread": True,
            }
        }
        super().setUp()

    def test_get_params_as_args(self):
        """Test that the parameters can be retrieved."""
        ipsi_args = list(self.model.ipsi.get_params(as_dict=False))
        contra_args = list(self.model.contra.get_params(as_dict=False))
        both_args = list(self.model.get_params(as_dict=False))
        self.assertEqual(ipsi_args, both_args)
        self.assertEqual(contra_args, both_args)

    def test_get_params_as_dict(self):
        """Test that the parameters can be retrieved."""
        ipsi_dict = self.model.ipsi.get_params()
        contra_dict = self.model.contra.get_params()
        both_dict = self.model.get_params()
        self.assertEqual(ipsi_dict, both_dict)
        self.assertEqual(contra_dict, both_dict)

    def test_set_params_as_args(self):
        """Test that the parameters can be set."""
        args_to_set = [self.rng.uniform() for _ in self.model.ipsi.get_params(as_dict=False)]
        self.model.set_params(*args_to_set)
        self.assertEqual(args_to_set, list(self.model.contra.get_params().values()))

    def test_set_params_as_dict(self):
        """Test that the parameters can be set via keyword arguments."""
        params_to_set = {k: self.rng.uniform() for k in self.model.contra.get_params()}
        self.model.set_params(**params_to_set)
        self.assertEqual(params_to_set, self.model.ipsi.get_params())


class LikelihoodTestCase(fixtures.BilateralModelMixin, fixtures.IgnoreWarningsTestCase):
    """Check that the (log-)likelihood is computed correctly."""

    def setUp(self):
        super().setUp()
        self.model.replace_all_modalities(fixtures.MODALITIES)
        self.load_patient_data()

    def test_compute_likelihood_twice(self):
        """Test that the likelihood is computed correctly."""
        first_llh = self.model.likelihood(log=True)
        second_llh = self.model.likelihood(log=True)
        self.assertEqual(first_llh, second_llh)


class RiskTestCase(fixtures.BilateralModelMixin, fixtures.IgnoreWarningsTestCase):
    """Check that the risk is computed correctly."""

    def setUp(self):
        super().setUp()
        self.model.replace_all_modalities(fixtures.MODALITIES)

    def create_random_diagnoses(self):
        """Create a random diagnosis for each modality and LNL."""
        diagnoses = {}

        for side in ["ipsi", "contra"]:
            diagnoses[side] = {}
            side_model = getattr(self.model, side)
            lnl_names = side_model.graph.lnls.keys()
            for modality in side_model.get_all_modalities():
                diagnoses[side][modality] = fixtures.create_random_pattern(lnl_names)

        return diagnoses

    def test_posterior_state_dist(self):
        """Test that the posterior state distribution is computed correctly."""
        num_states = len(self.model.ipsi.graph.state_list)
        random_parameters = self.create_random_params()
        random_diagnoses = self.create_random_diagnoses()

        posterior = self.model.posterior_state_dist(
            given_params=random_parameters,
            given_diagnoses=random_diagnoses,
        )
        self.assertEqual(posterior.shape, (num_states, num_states))
        self.assertEqual(posterior.dtype, float)
        self.assertTrue(np.isclose(posterior.sum(), 1.))

    def test_risk(self):
        """Test that the risk is computed correctly."""
        random_parameters = self.create_random_params()
        random_diagnoses = self.create_random_diagnoses()
        random_pattern = {
            "ipsi": fixtures.create_random_pattern(self.model.ipsi.graph.lnls.keys()),
            "contra": fixtures.create_random_pattern(self.model.contra.graph.lnls.keys()),
        }
        random_t_stage = self.rng.choice(["early", "late"])

        risk = self.model.risk(
            involvement=random_pattern,
            given_params=random_parameters,
            given_diagnoses=random_diagnoses,
            t_stage=random_t_stage,
        )
        self.assertLessEqual(risk, 1.)
        self.assertGreaterEqual(risk, 0.)


class DataGenerationTestCase(
    fixtures.BilateralModelMixin,
    fixtures.IgnoreWarningsTestCase,
):
    """Check the binary model's data generation method."""

    def setUp(self):
        super().setUp()
        self.model.replace_all_modalities(fixtures.MODALITIES)
        self.init_diag_time_dists(early="frozen", late="parametric")
        self.model.set_params(**self.create_random_params())

    def test_generate_data(self):
        """Check bilateral data generation."""
        dataset = self.model.draw_patients(
            num=10000,
            stage_dist=[0.5, 0.5],
            rng=self.rng,
        )

        for mod in self.model.get_all_modalities():
            self.assertIn(mod, dataset)
            for side in ["ipsi", "contra"]:
                self.assertIn(side, dataset[mod])
                for lnl in self.model.ipsi.graph.lnls.keys():
                    self.assertIn(lnl, dataset[mod][side])

        self.assertAlmostEqual(
            (dataset["tumor", "1", "t_stage"] == "early").mean(), 0.5,
            delta=0.02
        )
