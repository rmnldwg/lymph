"""Test the binary unilateral system."""
import unittest

import fixtures
import numpy as np

from lymph.graph import LymphNodeLevel, Tumor
from lymph.modalities import Pathological


class InitTestCase(fixtures.BinaryUnilateralModelMixin, unittest.TestCase):
    """Test the initialization of a binary model."""

    def test_num_nodes(self):
        """Check number of nodes initialized."""
        num_nodes = len(self.graph_dict)
        num_tumor = len({name for kind, name in self.graph_dict if kind == "tumor"})
        num_lnls = len({name for kind, name in self.graph_dict if kind == "lnl"})

        self.assertEqual(len(self.model.graph.nodes), num_nodes)
        self.assertEqual(len(self.model.graph.tumors), num_tumor)
        self.assertEqual(len(self.model.graph.lnls), num_lnls)

    def test_num_edges(self):
        """Check number of edges initialized."""
        num_edges = sum(len(receiving_nodes) for receiving_nodes in self.graph_dict.values())
        num_tumor_edges = sum(
            len(receiving_nodes) for (kind, _), receiving_nodes in self.graph_dict.items()
            if kind == "tumor"
        )
        num_lnl_edges = sum(
            len(receiving_nodes) for (kind, _), receiving_nodes in self.graph_dict.items()
            if kind == "lnl"
        )

        self.assertEqual(len(self.model.graph.edges), num_edges)
        self.assertEqual(len(self.model.graph.tumor_edges), num_tumor_edges)
        self.assertEqual(len(self.model.graph.lnl_edges), num_lnl_edges)
        self.assertEqual(len(self.model.graph.growth_edges), 0)

    def test_tumor(self):
        """Make sure the tumor has been initialized correctly."""
        tumor = self.model.graph.nodes["T"]
        state = tumor.state
        self.assertIsInstance(tumor, Tumor)
        self.assertListEqual(tumor.allowed_states, [state])

    def test_lnls(self):
        """Test they are all binary lymph node levels."""
        model_allowed_states = self.model.graph.allowed_states
        self.assertEqual(len(model_allowed_states), 2)

        for lnl in self.model.graph.lnls.values():
            self.assertIsInstance(lnl, LymphNodeLevel)
            self.assertTrue(lnl.is_binary)
            self.assertEqual(lnl.allowed_states, model_allowed_states)

    def test_tumor_to_lnl_edges(self):
        """Make sure the tumor to LNL edges have been initialized correctly."""
        tumor = self.model.graph.nodes["T"]
        receiving_lnls = self.graph_dict[("tumor", "T")]
        connecting_edge_names = [f"{tumor.name}_to_{lnl}" for lnl in receiving_lnls]

        for edge in self.model.graph.tumor_edges.values():
            self.assertEqual(edge.parent.name, "T")
            self.assertIn(edge.child.name, receiving_lnls)
            self.assertTrue(edge.is_tumor_spread)
            self.assertIn(edge.name, connecting_edge_names)


class DelegationTestCase(fixtures.BinaryUnilateralModelMixin, unittest.TestCase):
    """Test the delegation of parameters via the `DelegatorMixing`."""

    def test_delegation(self):
        """Make sure the specified attributes from graph are delegated upwards."""
        self.assertEqual(
            self.model.graph.is_binary,
            self.model.is_binary,
        )
        self.assertEqual(
            self.model.graph.is_trinary,
            self.model.is_trinary,
        )
        self.assertEqual(
            self.model.graph.get_state(),
            self.model.get_state(),
        )
        self.assertEqual(
            self.model.graph.lnls,
            self.model.lnls,
        )

    def test_set_state_delegation(self):
        """Check that the ``set_state`` method is also correctly delegated."""
        old_state = self.model.get_state()
        choice = [0,1]
        if self.model.is_trinary:
            choice.append(2)

        new_state = self.rng.choice(a=choice, size=len(old_state))
        self.model.set_state(*new_state)
        self.assertTrue(np.all(self.model.get_state() == new_state))
        self.assertTrue(np.all(self.model.graph.get_state() == new_state))

        new_state = self.rng.choice(a=choice, size=len(old_state))
        self.model.graph.set_state(*new_state)
        self.assertTrue(np.all(self.model.get_state() == new_state))
        self.assertTrue(np.all(self.model.graph.get_state() == new_state))


class ParameterAssignmentTestCase(fixtures.BinaryUnilateralModelMixin, unittest.TestCase):
    """Test the assignment of parameters in a binary model."""

    def test_params_assignment_via_lookup(self):
        """Make sure the spread parameters are assigned correctly."""
        params_to_set = self.create_random_params()
        edges_and_dists = self.model.graph.edges.copy()
        edges_and_dists.update(self.model.diag_time_dists)

        for param_name, value in params_to_set.items():
            name, type_ = param_name.rsplit("_", maxsplit=1)
            edges_and_dists[name].set_params(**{type_: value})
            self.assertEqual(
                edges_and_dists[name].get_params(type_),
                value,
            )

    def test_params_assignment_via_method(self):
        """Make sure the spread parameters are assigned correctly."""
        params_to_set = self.create_random_params()
        self.model.assign_params(**params_to_set)

        edges_and_dists = self.model.graph.edges.copy()
        edges_and_dists.update(self.model.diag_time_dists)

        for param_name, value in params_to_set.items():
            name, type_ = param_name.rsplit("_", maxsplit=1)
            self.assertEqual(
                edges_and_dists[name].get_params(type_),
                value,
            )

    def test_transition_matrix_deletion(self):
        """Check if the transition matrix gets deleted when a parameter is set."""
        first_lnl_name = list(self.model.graph.lnls.values())[0].name
        trans_mat = self.model.transition_matrix()
        self.model.graph.edges[f"T_to_{first_lnl_name}"].set_spread_prob(0.5)
        self.assertFalse(np.all(
            trans_mat == self.model.transition_matrix()
        ))


class TransitionMatrixTestCase(fixtures.BinaryUnilateralModelMixin, unittest.TestCase):
    """Test the generation of the transition matrix in a binary model."""

    def setUp(self):
        """Initialize a simple binary model."""
        super().setUp()
        self.model.assign_params(**self.create_random_params())

    def test_shape(self):
        """Make sure the transition matrix has the correct shape."""
        num_lnls = len({name for kind, name in self.graph_dict if kind == "lnl"})
        self.assertEqual(self.model.transition_matrix().shape, (2**num_lnls, 2**num_lnls))

    def test_is_probabilistic(self):
        """Make sure the rows of the transition matrix sum to one."""
        row_sums = np.sum(self.model.transition_matrix(), axis=1)
        self.assertTrue(np.allclose(row_sums, 1.))

    @staticmethod
    def is_recusively_upper_triangular(mat: np.ndarray) -> bool:
        """Return `True` is `mat` is recursively upper triangular."""
        if mat.shape == (1, 1):
            return True

        if not np.all(np.equal(np.triu(mat), mat)):
            return False

        half = mat.shape[0] // 2
        for i in [0, 1]:
            for j in [0, 1]:
                return TransitionMatrixTestCase.is_recusively_upper_triangular(
                    mat[i * half:(i + 1) * half, j * half:(j + 1) * half]
                )

    def test_is_recusively_upper_triangular(self) -> None:
        """Make sure the transition matrix is recursively upper triangular."""
        self.assertTrue(self.is_recusively_upper_triangular(self.model.transition_matrix()))


class ObservationMatrixTestCase(fixtures.BinaryUnilateralModelMixin, unittest.TestCase):
    """Test the generation of the observation matrix in a binary model."""

    def setUp(self):
        """Initialize a simple binary model."""
        super().setUp()
        self.model.modalities = fixtures.MODALITIES

    def test_shape(self):
        """Make sure the observation matrix has the correct shape."""
        num_lnls = len(self.model.graph.lnls)
        num_modalities = len(self.model.modalities)
        expected_shape = (2**num_lnls, 2**(num_lnls * num_modalities))
        self.assertEqual(self.model.observation_matrix().shape, expected_shape)

    def test_is_probabilistic(self):
        """Make sure the rows of the observation matrix sum to one."""
        row_sums = np.sum(self.model.observation_matrix(), axis=1)
        self.assertTrue(np.allclose(row_sums, 1.))


class PatientDataTestCase(fixtures.BinaryUnilateralModelMixin, unittest.TestCase):
    """Test loading the patient data."""

    def setUp(self):
        """Load patient data."""
        super().setUp()
        self.model.modalities = fixtures.MODALITIES
        self.init_diag_time_dists(early="frozen", late="parametric", foo="frozen")
        self.model.assign_params(**self.create_random_params())
        self.load_patient_data(filename="2021-usz-oropharynx.csv")

    def test_load_patient_data(self):
        """Make sure the patient data is loaded correctly."""
        self.assertEqual(len(self.model.patient_data), len(self.raw_data))
        self.assertRaises(
            ValueError, self.model.load_patient_data, self.raw_data, side="foo"
        )

    def test_t_stages(self):
        """Make sure all T-stages are present."""
        t_stages_in_data = self.model.patient_data["_model", "#" ,"t_stage"].unique()
        t_stages_in_diag_time_dists = self.model.diag_time_dists.keys()
        t_stages_in_model = list(self.model.t_stages)
        t_stages_intersection = set(t_stages_in_data).intersection(t_stages_in_diag_time_dists)

        self.assertNotIn("foo", t_stages_in_model)
        self.assertEqual(len(t_stages_in_diag_time_dists), 3)
        self.assertEqual(len(t_stages_intersection), 2)
        self.assertEqual(len(t_stages_intersection), len(t_stages_in_model))

        for t_stage in t_stages_in_model:
            self.assertIn(t_stage, t_stages_in_data)
            self.assertIn(t_stage, t_stages_in_diag_time_dists)

    def test_data_matrices(self):
        """Make sure the data matrices are generated correctly."""
        for t_stage in ["early", "late"]:
            has_t_stage = self.raw_data["tumor", "1", "t_stage"].isin({
                "early": [0,1,2],
                "late": [3,4],
            }[t_stage])
            data_matrix = self.model.data_matrices[t_stage]

            self.assertTrue(t_stage in self.model.data_matrices)
            self.assertEqual(
                data_matrix.shape[0],
                self.model.observation_matrix().shape[1],
            )
            self.assertEqual(
                data_matrix.shape[1],
                has_t_stage.sum(),
            )

    def test_diagnose_matrices(self):
        """Make sure the diagnose matrices are generated correctly."""
        for t_stage in ["early", "late"]:
            has_t_stage = self.raw_data["tumor", "1", "t_stage"].isin({
                "early": [0,1,2],
                "late": [3,4],
            }[t_stage])
            diagnose_matrix = self.model.diagnose_matrices[t_stage]

            self.assertTrue(t_stage in self.model.diagnose_matrices)
            self.assertEqual(
                diagnose_matrix.shape[0],
                self.model.transition_matrix().shape[1],
            )
            self.assertEqual(
                diagnose_matrix.shape[1],
                has_t_stage.sum(),
            )
            # some times, entries in the diagnose matrix are almost one, but just
            # slightly larger. That's why we also have to have the `isclose` here.
            self.assertTrue(np.all(
                np.isclose(diagnose_matrix, 1.)
                | np.less_equal(diagnose_matrix, 1.)
            ))


class LikelihoodTestCase(fixtures.BinaryUnilateralModelMixin, unittest.TestCase):
    """Test the likelihood of a model."""

    def setUp(self):
        """Load patient data."""
        super().setUp()
        self.model.modalities = fixtures.MODALITIES
        self.init_diag_time_dists(early="frozen", late="parametric")
        self.model.assign_params(**self.create_random_params())
        self.load_patient_data(filename="2021-usz-oropharynx.csv")

    def test_log_likelihood_smaller_zero(self):
        """Make sure the log-likelihood is smaller than zero."""
        likelihood = self.model.likelihood(log=True, mode="HMM")
        self.assertLess(likelihood, 0.)

    def test_likelihood_invalid_params_isinf(self):
        """Make sure the likelihood is `-np.inf` for invalid parameters."""
        random_params = self.create_random_params()
        for name in random_params:
            random_params[name] += 1.
        likelihood = self.model.likelihood(
            given_param_kwargs=random_params,
            log=True,
            mode="HMM",
        )
        self.assertEqual(likelihood, -np.inf)


class RiskTestCase(fixtures.BinaryUnilateralModelMixin, unittest.TestCase):
    """Test anything related to the risk computation."""

    def setUp(self):
        """Load params."""
        super().setUp()
        self.model.modalities = fixtures.MODALITIES
        self.init_diag_time_dists(early="frozen", late="parametric")
        self.model.assign_params(**self.create_random_params())

    def create_random_diagnoses(self):
        """Create a random diagnosis for each modality and LNL."""
        lnl_names = list(self.model.graph.lnls.keys())
        diagnoses = {}

        for modality in self.model.modalities:
            diagnoses[modality] = fixtures.create_random_pattern(lnl_names)

        return diagnoses

    def test_comp_diagnose_encoding(self):
        """Check computation of one-hot encoding of diagnoses."""
        random_diagnoses = self.create_random_diagnoses()
        num_lnls, num_mods = len(self.model.graph.lnls), len(self.model.modalities)
        num_posible_diagnoses = 2**(num_lnls * num_mods)

        diagnose_encoding = self.model.comp_diagnose_encoding(random_diagnoses)
        self.assertEqual(diagnose_encoding.shape, (num_posible_diagnoses,))
        self.assertEqual(diagnose_encoding.dtype, bool)

    def test_posterior_state_dist(self):
        """Make sure the posterior state dist is correctly computed."""
        posterior_state_dist = self.model.comp_posterior_state_dist(
            given_param_kwargs=self.create_random_params(),
            given_diagnoses=self.create_random_diagnoses(),
            t_stage=self.rng.choice(["early", "late"]),
        )
        self.assertEqual(posterior_state_dist.shape, (2**len(self.model.graph.lnls),))
        self.assertEqual(posterior_state_dist.dtype, float)
        self.assertTrue(np.isclose(np.sum(posterior_state_dist), 1.))

    def test_risk(self):
        """Make sure the risk is correctly computed."""
        random_pattern = fixtures.create_random_pattern(self.model.graph.lnls.keys())
        random_diagnoses = self.create_random_diagnoses()
        random_t_stage = self.rng.choice(["early", "late"])
        random_params = self.create_random_params()

        risk = self.model.risk(
            involvement=random_pattern,
            given_param_kwargs=random_params,
            given_diagnoses=random_diagnoses,
            t_stage=random_t_stage,
        )
        self.assertEqual(risk.dtype, float)
        self.assertGreaterEqual(risk, 0.)
        self.assertLessEqual(risk, 1.)


class DataGenerationTestCase(fixtures.BinaryUnilateralModelMixin, unittest.TestCase):
    """Check the data generation utilities."""

    def setUp(self):
        """Load params."""
        super().setUp()
        self.model.modalities = fixtures.MODALITIES
        self.init_diag_time_dists(early="frozen", late="parametric")
        self.model.assign_params(**self.create_random_params())

    def test_generate_early_patients(self):
        """Check that generating only early T-stage patients works."""
        early_patients = self.model.draw_patients(
            num=100,
            stage_dist=[1., 0.],
            rng=self.rng,
        )
        self.assertEqual(len(early_patients), 100)
        self.assertEqual(sum(early_patients["tumor", "1", "t_stage"] == "early"), 100)
        self.assertIn(("CT", "ipsi", "II"), early_patients.columns)
        self.assertIn(("FNA", "ipsi", "III"), early_patients.columns)

    def test_generate_late_patients(self):
        """Check that generating only late T-stage patients works."""
        late_patients = self.model.draw_patients(
            num=100,
            stage_dist=[0., 1.],
            rng=self.rng,
        )
        self.assertEqual(len(late_patients), 100)
        self.assertEqual(sum(late_patients["tumor", "1", "t_stage"] == "late"), 100)
        self.assertIn(("CT", "ipsi", "II"), late_patients.columns)
        self.assertIn(("FNA", "ipsi", "III"), late_patients.columns)

    def test_distribution_of_patients(self):
        """Check that the distribution of LNL involvement is correct."""
        # set spread params all to 0
        for lnl_edge in self.model.graph.lnl_edges.values():
            lnl_edge.set_spread_prob(0.)

        # make all patients diagnosed after exactly one time-step
        self.model.diag_time_dists["early"] = [0,1,0,0,0,0,0,0,0,0,0]

        # assign only one pathology modality
        self.model.modalities = {"tmp": Pathological(specificity=1., sensitivity=1.)}

        # extract the tumor spread parameters
        params = self.model.get_params(as_dict=True)
        params = {
            key.replace("T_to_", "").replace("_spread", ""): value
            for key, value in params.items()
            if "T_to_" in key
        }

        # draw large enough amount of patients
        patients = self.model.draw_patients(
            num=10000,
            stage_dist=[1., 0.],
            rng=self.rng,
        )

        # check that the distribution of LNL involvement matches tumor spread params
        for lnl, expected_mean in params.items():
            actual_mean = patients[("tmp", "ipsi", lnl)].mean()
            self.assertAlmostEqual(actual_mean, expected_mean, delta=0.02)
