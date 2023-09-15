from __future__ import annotations

import logging
import warnings
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from lymph import graph, matrix, modalities, models
from lymph.helper import (
    AbstractLookupDict,
    DelegatorMixin,
    DiagnoseType,
    PatternType,
    early_late_mapping,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)



def create_property_sync_callback(
    names: list[str],
    this: graph.Edge,
    other: graph.Edge,
) -> callable:
    """Return func to sync property values whose name is in ``names`` btw two edges.

    The returned function is meant to be added to the list of callbacks of the
    :py:class:`Edge` class, such that two edges in a mirrored pair of graphs are kept
    in sync.
    """
    def sync():
        # We must set the value of `this` property via the private name, otherwise
        # we would trigger the setter's callbacks and may end up in an infinite loop.
        for name in names:
            private_name = f"_{name}"
            setattr(other, private_name, getattr(this, name))

    logger.debug(f"Created sync callback for properties {names} of {this.name} edge.")
    return sync


def init_edge_sync(
    property_names: list[str],
    this_edge_list: list[graph.Edge],
    other_edge_list: list[graph.Edge],
) -> None:
    """Initialize the callbacks to sync properties btw. Edges.

    Implementing this as a separate method allows a user in theory to initialize
    an arbitrary kind of symmetry between the two sides of the neck.
    """
    this_edge_names = [e.name for e in this_edge_list]
    other_edge_names = [e.name for e in other_edge_list]

    for edge_name in set(this_edge_names).intersection(other_edge_names):
        this_edge = this_edge_list[this_edge_names.index(edge_name)]
        other_edge = other_edge_list[other_edge_names.index(edge_name)]

        this_edge.trigger_callbacks.append(
            create_property_sync_callback(
                names=property_names,
                this=this_edge,
                other=other_edge,
            )
        )
        other_edge.trigger_callbacks.append(
            create_property_sync_callback(
                names=property_names,
                this=other_edge,
                other=this_edge,
            )
        )


def init_dict_sync(
    this: AbstractLookupDict,
    other: AbstractLookupDict,
) -> None:
    """Add callback to ``this`` to sync with ``other``."""
    def sync():
        other.clear()
        other.update(this)

    this.trigger_callbacks.append(sync)


class Bilateral(DelegatorMixin):
    """Class that models metastatic progression in a bilateral lymphatic system.

    This is achieved by creating two instances of the
    :py:class:`~lymph.models.Unilateral` model, one for the ipsi- and one for the
    contralateral side of the neck. The two sides are assumed to be independent of each
    other, given the diagnose time over which we marginalize.

    See Also:
        :py:class:`~lymph.models.Unilateral`
            Two instances of this class are created as attributes.
        :py:class:`~lymph.descriptors.Distribution`
            A class to store fixed and parametric distributions over diagnose times.
    """
    def __init__(
        self,
        graph_dict: dict[tuple[str], list[str]],
        tumor_spread_symmetric: bool = False,
        lnl_spread_symmetric: bool = True,
        modalities_symmetric: bool = True,
        unilateral_kwargs: dict[str, Any] | None = None,
        **_kwargs,
    ) -> None:
        """Initialize both sides of the neck as :py:class:`~lymph.models.Unilateral`.

        The ``graph_dict`` is a dictionary of tuples as keys and lists of strings as
        values. It is passed to both :py:class:`~lymph.models.Unilateral` instances,
        which in turn pass it to the :py:class:`~lymph.graph.Representation` class that
        stores the graph.

        The ``tumor_spread_symmetric`` and ``lnl_spread_symmetric`` arguments determine
        which parameters are shared between the two sides of the neck. If
        ``tumor_spread_symmetric`` is ``True``, the spread probabilities from the
        tumor(s) to the LNLs are shared. If ``lnl_spread_symmetric`` is ``True``, the
        spread probabilities between the LNLs are shared.

        The ``unilateral_kwargs`` are passed to both instances of the unilateral model.
        """
        super().__init__()

        # TODO: Implement asymmetric model. This should be relatively straightforward,
        #       since the transition matrices of the two sides do not need to have the
        #       same shape. The only thing that needs to be changed is the constructor
        #       of this class, where the two instances of the unilateral model are
        #       created.
        if unilateral_kwargs is None:
            unilateral_kwargs = {}

        self.ipsi   = models.Unilateral(graph_dict=graph_dict, **unilateral_kwargs)
        self.contra = models.Unilateral(graph_dict=graph_dict, **unilateral_kwargs)

        self.tumor_spread_symmetric  = tumor_spread_symmetric
        self.lnl_spread_symmetric = lnl_spread_symmetric
        self.modalities_symmetric = modalities_symmetric

        property_names = ["spread_prob"]
        if self.ipsi.graph.is_trinary:
            property_names.append("micro_mod")

        if self.tumor_spread_symmetric:
            init_edge_sync(
                property_names,
                list(self.ipsi.graph.tumor_edges.values()),
                list(self.contra.graph.tumor_edges.values()),
            )

        if self.lnl_spread_symmetric:
            ipsi_edges = (
                list(self.ipsi.graph.lnl_edges.values())
                + list(self.ipsi.graph.growth_edges.values())
            )
            contra_edges = (
                list(self.contra.graph.lnl_edges.values())
                + list(self.contra.graph.growth_edges.values())
            )
            init_edge_sync(property_names, ipsi_edges, contra_edges)

        delegated_attrs = [
            "max_time", "t_stages", "diag_time_dists",
            "is_binary", "is_trinary",
        ]

        init_dict_sync(
            this=self.ipsi.diag_time_dists,
            other=self.contra.diag_time_dists,
        )

        if self.modalities_symmetric:
            delegated_attrs.append("modalities")
            init_dict_sync(
                this=self.ipsi.modalities,
                other=self.contra.modalities,
            )

        self.init_delegation(ipsi=delegated_attrs)
        self.contra.diag_time_dists = self.diag_time_dists


    def get_params(
        self,
        param: str | None = None,
        as_dict: bool = False,
        nested: bool = False,
    ) -> float | Iterable[float] | dict[str, float] | dict[str, dict[str, float]]:
        """Return the parameters of the model.

        If ``nested`` is ``True``, the parameters of the two sides of the neck are
        returned as a nested dictionary in addition to one dictionary storing the
        parameters of the parametric distributions for marginalizing over diagnose
        times. Otherwise, the parameters are returned as a flat dictionary, with the
        keys prefixed by ``"ipsi_"`` or ``"contra_"``. The parameters of the parametric
        distributions are only prefixed by their corresponding T-stage, e.g.
        ``"early_p"``.

        If ``as_dict`` is ``True``, the parameters are returned as a dictionary. If
        ``param`` is not ``None``, only the value of the parameter with that name is
        returned. Otherwise, all parameters are returned as a dictionary or a list.

        Note:
            The arguments ``as_dict`` and ``nested`` are ignored if ``param`` is not
            ``None``. Also, ``nested`` is ignored if ``as_dict`` is ``False``.
        """
        ipsi_params = self.ipsi.get_params(as_dict=True, with_dists=False)
        contra_params = self.contra.get_params(as_dict=True, with_dists=False)
        dist_params = self.ipsi.get_params(as_dict=True, with_edges=False)

        if nested and as_dict and param is None:
            return {
                "ipsi": ipsi_params,
                "contra": contra_params,
                "diag_time_dists": dist_params,
            }

        params = {f"ipsi_{k}": v for k, v in ipsi_params.items()}
        params.update({f"contra_{k}": v for k, v in contra_params.items()})
        params.update(dist_params)

        if param is not None:
            return params[param]

        return params if as_dict else params.values()



    def assign_params(
        self,
        *new_params_args,
        **new_params_kwargs,
    ) -> tuple[Iterator[float, dict[str, float]]]:
        """Assign new parameters to the model.

        This works almost exactly as the unilateral model's
        :py:meth:`~lymph.models.Unilateral.assign_params` method. However, this one
        allows the user to set the parameters of individual sides of the neck by
        prefixing the parameter name with ``"ipsi_"`` or ``"contra_"``. This is
        necessary for parameters that are not symmetric between the two sides of the
        neck. For symmetric parameters, the prefix is not needed as they are directly
        sent to the ipsilateral side, which then triggers a sync callback.

        Note:
            When setting the parameters via positional arguments, the order is
            important. The first ``len(self.ipsi.get_params(as_dict=True))`` arguments
            are passed to the ipsilateral side, the remaining ones to the contralateral
            side.
        """
        ipsi_kwargs, contra_kwargs, general_kwargs = {}, {}, {}
        for key, value in new_params_kwargs.items():
            if "ipsi_" in key:
                ipsi_kwargs[key.replace("ipsi_", "")] = value
            elif "contra_" in key:
                contra_kwargs[key.replace("contra_", "")] = value
            else:
                general_kwargs[key] = value

        remaining_args, remainings_kwargs = self.ipsi.assign_params(
            *new_params_args, **ipsi_kwargs, **general_kwargs
        )
        remaining_args, remainings_kwargs = self.contra.assign_params(
            *remaining_args, **contra_kwargs, **remainings_kwargs
        )
        return remaining_args, remainings_kwargs


    @property
    def modalities(self) -> modalities.ModalitiesUserDict:
        """Return the set diagnostic modalities of the model.

        See Also:
            :py:attr:`lymph.models.Unilateral.modalities`
                The corresponding unilateral attribute.
            :py:class:`~lymph.descriptors.ModalitiesUserDict`
                The implementation of the descriptor class.
        """
        if not self.modalities_symmetric:
            raise AttributeError(
                "The modalities are not symmetric. Please access them via the "
                "`ipsi` or `contra` attributes."
            )
        return self.ipsi.modalities

    @modalities.setter
    def modalities(self, new_modalities) -> None:
        """Set the diagnostic modalities of the model."""
        if not self.modalities_symmetric:
            raise AttributeError(
                "The modalities are not symmetric. Please set them via the "
                "`ipsi` or `contra` attributes."
            )
        self.ipsi.modalities = new_modalities


    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        mapping: callable = early_late_mapping,
    ) -> None:
        """Load patient data into the model.

        This amounts to calling the :py:meth:`~lymph.models.Unilateral.load_patient_data`
        method of both sides of the neck.
        """
        self.ipsi.load_patient_data(patient_data, "ipsi", mapping)
        self.contra.load_patient_data(patient_data, "contra", mapping)


    def comp_joint_state_dist(
        self,
        t_stage: str = "early",
        mode: str = "HMM",
    ) -> np.ndarray:
        """Compute the joint distribution over the ipsi- & contralateral hidden states.

        This computes the state distributions of both sides and returns their outer
        product. In case ``mode`` is ``"HMM"`` (default), the state distributions are
        first marginalized over the diagnose time distribtions of the respective
        ``t_stage``.

        See Also:
            :py:meth:`lymph.models.Unilateral.comp_state_dist`
                The corresponding unilateral function.
        """
        if mode == "HMM":
            ipsi_state_evo = self.ipsi.comp_dist_evolution()
            contra_state_evo = self.contra.comp_dist_evolution()
            time_marg_matrix = np.diag(self.diag_time_dists[t_stage].distribution)

            result = (
                ipsi_state_evo.T
                @ time_marg_matrix
                @ contra_state_evo
            )

        elif mode == "BN":
            ipsi_state_dist = self.ipsi.comp_state_dist(mode=mode)
            contra_state_dist = self.contra.comp_state_dist(mode=mode)

            result = np.outer(ipsi_state_dist, contra_state_dist)

        else:
            raise ValueError(f"Unknown mode '{mode}'.")

        return result


    def comp_joint_obs_dist(
        self,
        t_stage: str = "early",
        mode: str = "HMM",
    ) -> np.ndarray:
        """Compute the joint distribution over the ipsi- & contralateral observations.

        See Also:
            :py:meth:`lymph.models.Unilateral.comp_obs_dist`
                The corresponding unilateral function.
        """
        joint_state_dist = self.comp_joint_state_dist(t_stage=t_stage, mode=mode)
        return (
            self.ipsi.observation_matrix.T
            @ joint_state_dist
            @ self.contra.observation_matrix
        )


    def _bn_likelihood(self, log: bool = True) -> float:
        """Compute the BN likelihood of data, using the stored params."""
        llh = 0. if log else 1.

        joint_state_dist = self.comp_joint_state_dist(mode="BN")
        joint_diagnose_dist = np.sum(
            self.ipsi.stacked_diagnose_matrix
            * (joint_state_dist @ self.contra.stacked_diagnose_matrix),
            axis=0,
        )

        if log:
            llh += np.sum(np.log(joint_diagnose_dist))
        else:
            llh *= np.prod(joint_diagnose_dist)
        return llh


    def _hmm_likelihood(self, log: bool = True) -> float:
        """Compute the HMM likelihood of data, using the stored params."""
        llh = 0. if log else 1.

        ipsi_dist_evo = self.ipsi.comp_dist_evolution()
        contra_dist_evo = self.contra.comp_dist_evolution()

        for stage in self.t_stages:
            diag_time_matrix = np.diag(self.diag_time_dists[stage].distribution)

            # Note that I am not using the `comp_joint_state_dist` method here, since
            # that would recompute the state dist evolution for each T-stage.
            joint_state_dist = (
                ipsi_dist_evo.T
                @ diag_time_matrix
                @ contra_dist_evo
            )
            # the computation below is a trick to make the computation fatser:
            # What we want to compute is the sum over the diagonal of the matrix
            # product of the ipsi diagnose matrix with the joint state distribution
            # and the contra diagnose matrix.
            # Source: https://stackoverflow.com/a/18854776
            joint_diagnose_dist = np.sum(
                self.ipsi.diagnose_matrices[stage]
                * (joint_state_dist @ self.contra.diagnose_matrices[stage]),
                axis=0,
            )

            if log:
                llh += np.sum(np.log(joint_diagnose_dist))
            else:
                llh *= np.prod(joint_diagnose_dist)

        return llh


    def likelihood(
        self,
        data: pd.DataFrame | None = None,
        given_param_args: Iterable[float] | None = None,
        given_param_kwargs: dict[str, float] | None = None,
        load_data_kwargs: dict[str, Any] | None = None,
        log: bool = True,
        mode: str = "HMM"
    ):
        """Compute the (log-)likelihood of the ``data`` given the model (and params).

        If the ``data`` is not provided, the previously loaded data is used. One may
        specify additional ``load_data_kwargs`` to pass to the
        :py:meth:`~load_patient_data` method when loading the data.

        The parameters of the model can be set via ``given_param_args`` and
        ``given_param_kwargs``. Both arguments are used to call the
        :py:meth:`~assign_params` method. If the parameters are not provided, the
        previously assigned parameters are used.

        Returns the log-likelihood if ``log`` is set to ``True``. The ``mode`` parameter
        determines whether the likelihood is computed for the hidden Markov model
        (``"HMM"``) or the Bayesian network (``"BN"``).

        Note:
            The computation is much faster if no parameters are given, since then the
            transition matrix does not need to be recomputed.

        See Also:
            :py:meth:`lymph.models.Unilateral.likelihood`
                The corresponding unilateral function.
        """
        if data is not None:
            if load_data_kwargs is None:
                load_data_kwargs = {}
            self.load_patient_data(data, **load_data_kwargs)

        if given_param_args is None:
            given_param_args = []

        if given_param_kwargs is None:
            given_param_kwargs = {}

        try:
            # all functions and methods called here should raise a ValueError if the
            # given parameters are invalid...
            self.assign_params(*given_param_args, **given_param_kwargs)
        except ValueError:
            return -np.inf if log else 0.

        return self._hmm_likelihood(log) if mode == "HMM" else self._bn_likelihood(log)


    def comp_posterior_joint_state_dist(
        self,
        given_param_args: Iterable[float] | None = None,
        given_param_kwargs: dict[str, float] | None = None,
        given_diagnoses: dict[str, DiagnoseType] | None = None,
        t_stage: str | int = "early",
        mode: str = "HMM",
    ) -> np.ndarray:
        """Compute joint post. dist. over ipsi & contra states, ``given_diagnoses``.

        The ``given_diagnoses`` is a dictionary storing a :py:class:`DiagnoseType` for
        the ``"ipsi"`` and ``"contra"`` side of the neck.

        Essentially, this is the risk for any possible combination of ipsi- and
        contralateral involvement, given the provided diagnoses.

        Note:
            The computation is much faster if no parameters are given, since then the
            transition matrix does not need to be recomputed.

        See Also:
            :py:meth:`lymph.models.Unilateral.comp_posterior_state_dist`
        """
        if given_param_args is None:
            given_param_args = []

        if given_param_kwargs is None:
            given_param_kwargs = {}

        self.assign_params(*given_param_args, **given_param_kwargs)

        if given_diagnoses is None:
            given_diagnoses = {}

        diagnose_given_state = {}
        for side in ["ipsi", "contra"]:
            diagnose_encoding = getattr(self, side).comp_diagnose_encoding(
                given_diagnoses.get(side, {})
            )
            observation_matrix = getattr(self, side).observation_matrix
            # vector with P(Z=z|X) for each state X. A data matrix for one "patient"
            diagnose_given_state[side] = diagnose_encoding @ observation_matrix

        joint_state_dist = self.comp_joint_state_dist(t_stage=t_stage, mode=mode)
        # matrix with P(Zi=zi,Zc=zc|Xi,Xc) * P(Xi,Xc) for all states Xi,Xc.
        joint_diagnose_and_state = (
            diagnose_given_state["ipsi"].T
            * joint_state_dist
            * diagnose_given_state["contra"]
        )
        # Following Bayes' theorem, this is P(Xi,Xc|Zi=zi,Zc=zc) which is given by
        # P(Zi=zi,Zc=zc|Xi,Xc) * P(Xi,Xc) / P(Zi=zi,Zc=zc)
        return joint_diagnose_and_state / np.sum(joint_diagnose_and_state)


    def risk(
        self,
        involvement: PatternType | None = None,
        given_param_args: Iterable[float] | None = None,
        given_param_kwargs: dict[str, float] | None = None,
        given_diagnoses: dict[str, DiagnoseType] | None = None,
        t_stage: str = "early",
        mode: str = "HMM",
    ) -> float:
        """Compute risk of an ``involvement`` pattern, given parameters and diagnoses.

        The parameters can be set via the ``given_param_args`` and
        ``given_param_kwargs``, both of which are passed to the
        :py:meth:`~assign_params` method. The ``given_diagnoses`` must be a dictionary
        mapping the side of the neck to a :py:class:`DiagnoseType`.

        Note:
            The computation is much faster if no parameters are given, since then the
            transition matrix does not need to be recomputed.

        See Also:
            :py:meth:`lymph.models.Unilateral.risk`
                The unilateral method for computing the risk of an involvment pattern.
            :py:meth:`lymph.models.Bilateral.comp_posterior_joint_state_dist`
                This method computes the joint distribution over ipsi- and
                contralateral states, given the parameters and diagnoses. The risk then
                only marginalizes over the states that match the involvement pattern.
        """
        # TODO: test this method
        posterior_state_probs = self.comp_posterior_joint_state_dist(
            given_param_args=given_param_args,
            given_param_kwargs=given_param_kwargs,
            given_diagnoses=given_diagnoses,
            t_stage=t_stage,
            mode=mode,
        )

        if involvement is None:
            return posterior_state_probs

        marginalize_over_states = {}
        for side in ["ipsi", "contra"]:
            marginalize_over_states[side] = matrix.compute_encoding(
                lnls=[lnl.name for lnl in self.graph.lnls],
                pattern=involvement[side],
            )
        return (
            marginalize_over_states["ipsi"]
            @ posterior_state_probs
            @ marginalize_over_states["contra"]
        )


    def generate_dataset(
        self,
        num_patients: int,
        stage_dist: dict[str, float],
    ) -> pd.DataFrame:
        """Generate/sample a pandas :class:`DataFrame` from the defined network.

        Args:
            num_patients: Number of patients to generate.
            stage_dist: Probability to find a patient in a certain T-stage.
        """
        # TODO: check if this still works
        drawn_t_stages, drawn_diag_times = self.diag_time_dists.draw(
            dist=stage_dist, size=num_patients
        )

        drawn_obs_ipsi = self.ipsi._draw_patient_diagnoses(drawn_diag_times)
        drawn_obs_contra = self.contra._draw_patient_diagnoses(drawn_diag_times)
        drawn_obs = np.concatenate([drawn_obs_ipsi, drawn_obs_contra], axis=1)

        # construct MultiIndex for dataset from stored modalities
        sides = ["ipsi", "contra"]
        modalities = list(self.modalities.keys())
        lnl_names = [lnl.name for lnl in self.ipsi.graph._lnls]
        multi_cols = pd.MultiIndex.from_product([sides, modalities, lnl_names])

        # create DataFrame
        dataset = pd.DataFrame(drawn_obs, columns=multi_cols)
        dataset = dataset.reorder_levels(order=[1, 0, 2], axis="columns")
        dataset = dataset.sort_index(axis="columns", level=0)
        dataset[('info', 'tumor', 't_stage')] = drawn_t_stages

        return dataset
