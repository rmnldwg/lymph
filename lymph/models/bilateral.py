from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from lymph import graph, models
from lymph.descriptors import modalities
from lymph.helper import DelegatorMixin, early_late_mapping

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)



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
            setattr(this, private_name, getattr(other, name))

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


def create_modality_sync_callback(
    this: modalities.ModalitiesUserDict,
    other: modalities.ModalitiesUserDict,
) -> None:
    """Create a callback func to synchronizes the modalities of ``this`` and ``other``.

    The callback goes through the keys of ``this`` and check if any of the modalities
    is missing in ``other``. If so, it adds the missing modality to ``other`` and
    initializes it with the same values as in ``this``.

    It also checks if there are any keys present in ``other`` that are not in ``this``.
    In that case, the respective item is removed from ``other``.

    Note:
        This is a one-way sync. Meaning that any changes to the modalities of ``other``
        are not reflected in ``this`` and get overwritten the next time this callback
        is triggered.

        But this should not be a problem, since for a :py:class:`Bilateral` model with
        symmetric modalities, only the modalities of the ipsilateral side should be
        accessed directly.
    """
    def sync():
        for modality in this.keys():
            if modality not in other:
                other[modality] = this[modality]

        for modality in other.keys():
            if modality not in this:
                del other[modality]

    return sync



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
                self.ipsi.graph.tumor_edges,
                self.contra.graph.tumor_edges,
            )

        if self.lnl_spread_symmetric:
            init_edge_sync(
                property_names,
                self.ipsi.graph.lnl_edges,
                self.contra.graph.lnl_edges,
            )

        if self.modalities_symmetric:
            self.modalities = self.ipsi.modalities
            self.ipsi.modalities.trigger_callbacks.append(
                create_modality_sync_callback(
                    this=self.ipsi.modalities,
                    other=self.contra.modalities,
                )
            )

        self.diag_time_dists = self.ipsi.diag_time_dists

        self.init_delegation(
            ipsi=["max_time", "t_stages", "is_binary", "is_trinary"],
        )


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


    def _likelihood(
        self,
        log: bool = True
    ) -> float:
        """Compute the (log-)likelihood of data, using the stored params."""
        # TODO: Continue here.
        stored_t_stages = set(self.ipsi.diagnose_matrices.keys())
        provided_t_stages = set(self.ipsi.diag_time_dists.keys())
        t_stages = list(stored_t_stages.intersection(provided_t_stages))

        max_t = self.diag_time_dists.max_t
        state_probs = {}
        state_probs["ipsi"] = self.ipsi._evolve(t_last=max_t)
        state_probs["contra"] = self.contra._evolve(t_last=max_t)

        llh = 0. if log else 1.
        for stage in t_stages:
            joint_state_probs = (
                state_probs["ipsi"].T
                @ np.diag(self.ipsi.diag_time_dists[stage].pmf)
                @ state_probs["contra"]
            )
            p = np.sum(
                self.ipsi.diagnose_matrices[stage]
                * (joint_state_probs
                    @ self.contra.diagnose_matrices[stage]),
                axis=0
            )
            if log:
                llh += np.sum(np.log(p))
            else:
                llh *= np.prod(p)

        return llh


    def likelihood(
        self,
        data: pd.DataFrame | None = None,
        given_params: np.ndarray | None = None,
        log: bool = True,
    ):
        """Compute log-likelihood of (already stored) data, given the spread
        probabilities and either a discrete diagnose time or a distribution to
        use for marginalization over diagnose times.

        Args:
            data: Table with rows of patients and columns of per-LNL involvment. See
                :meth:`load_data` for more details on how this should look like.

            given_params: The likelihood is a function of these parameters. They mainly
                consist of the :attr:`spread_probs` of the model. Any excess parameters
                will be used to update the parametrized distributions used for
                marginalizing over the diagnose times (see :attr:`diag_time_dists`).

            log: When ``True``, the log-likelihood is returned.

        Returns:
            The log-likelihood :math:`\\log{p(D \\mid \\theta)}` where :math:`D`
            is the data and :math:`\\theta` is the tuple of spread probabilities
            and diagnose times or distributions over diagnose times.

        See Also:
            :attr:`spread_probs`: Property for getting and setting the spread
            probabilities, of which a lymphatic network has as many as it has
            :class:`Edge` instances (in case no symmetries apply).

            :meth:`Unilateral.likelihood`: The (log-)likelihood function of
            the unilateral system.
        """
        if data is not None:
            self.patient_data = data

        if given_params is None:
            return self._likelihood(log)

        try:
            self.check_and_assign(given_params)
        except ValueError:
            return -np.inf if log else 0.

        return self._likelihood(log)


    def risk(
        self,
        involvement: dict | None = None,
        given_params: np.ndarray | None = None,
        given_diagnoses: dict | None = None,
        t_stage: str = "early",
        **_kwargs,
    ) -> float:
        """Compute risk of ipsi- & contralateral involvement given specific (but
        potentially incomplete) diagnoses for each side of the neck.

        Args:
            involvement: Nested dictionary that can have keys ``"ipsi"`` and
                ``"contra"``, indicating the respective side's involvement patterns
                that we're interested in. The corresponding values are dictionaries as
                the :class:`Unilateral` model expects them.

            given_params: The risk is a function of these parameters. They mainly
                consist of the :attr:`spread_probs` of the model. Any excess parameters
                will be used to update the parametrized distributions used for
                marginalizing over the diagnose times (see :attr:`diag_time_dists`).

            given_diagnoses: Nested dictionary with keys of diagnostic modalities and
                the values are dictionaries of the same format as the ``involvement``
                arguments.

            t_stage: The T-stage for which the risk should be computed. The attribute
                :attr:`diag_time_dists` must have a distribution for marginalizing
                over diagnose times stored for this T-stage.

        See Also:
            :meth:`Unilateral.risk`: The risk function for only one-sided models.
        """
        if given_params is not None:
            self.check_and_assign(given_params)

        if given_diagnoses is None:
            first_modality = list(self.modalities.keys())[0]
            given_diagnoses = {first_modality: {}}

        for val in given_diagnoses.values():
            if "ipsi" not in val:
                val["ipsi"] = {}
            if "contra" not in val:
                val["contra"] = {}

        diagnose_probs = {}   # vectors containing P(Z=z|X) for respective side
        state_probs = {}      # matrices containing P(X|t) for each side

        for side in ["ipsi", "contra"]:
            side_model = getattr(self, side)
            diagnose_probs[side] = np.zeros(shape=len(side_model.state_list))
            side_diagnose = {mod: diag[side] for mod,diag in given_diagnoses.items()}
            for i,state in enumerate(side_model.state_list):
                side_model.state = state
                diagnose_probs[side][i] = side_model.comp_diagnose_prob(side_diagnose)

            max_t = self.diag_time_dists.max_t
            state_probs[side] = self.system[side]._evolve(t_last=max_t)

        # time-prior in diagnoal matrix form
        time_marg_matrix = np.diag(self.ipsi.diag_time_dists[t_stage].pmf)

        # joint probability P(Xi,Xc) (marginalized over time). Acts as prior
        # for p( Di,Dc | Xi,Xc ) and should be a 2D matrix.
        marg_state_probs = (
            state_probs["ipsi"].T @ time_marg_matrix @ state_probs["contra"]
        )

        # joint probability P(Di=di,Dc=dc, Xi,Xc) of all hidden states and the
        # provided diagnoses
        joint_diag_state = np.einsum(
            "i,ij,j->ij",
            diagnose_probs["ipsi"], marg_state_probs, diagnose_probs["contra"],
        )
        # marginalized probability P(Di=di, Dc=dc)
        marg_diagnose_prob = np.sum(joint_diag_state)

        # P(Xi,Xc | Di=di,Dc=dc)
        post_state_probs = joint_diag_state / marg_diagnose_prob

        if involvement is None:
            return post_state_probs
        if "ipsi" not in involvement:
            involvement["ipsi"] = {}
        if "contra" not in involvement:
            involvement["contra"] = {}

        marg_states = {}   # vectors marginalizing over only the states we care about
        for side in ["ipsi", "contra"]:
            if isinstance(involvement[side], dict):
                involvement[side] = np.array(
                    [involvement[side].get(lnl.name, None) for lnl in side_model.lnls]
                )
            else:
                involvement[side] = np.array(involvement[side])

            side_model = getattr(self, side)
            marg_states[side] = np.zeros(shape=len(side_model.state_list), dtype=bool)
            for i,state in enumerate(side_model.state_list):
                marg_states[side][i] = np.all(np.equal(
                    involvement[side], state,
                    where=(involvement[side] != None),
                    out=np.ones_like(state, dtype=bool)
                ))

        return marg_states["ipsi"] @ post_state_probs @ marg_states["contra"]



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
