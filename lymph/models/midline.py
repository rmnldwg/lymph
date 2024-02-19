from __future__ import annotations

import logging
import warnings
from argparse import OPTIONAL
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from lymph import graph, modalities, models
from lymph.helper import (
    AbstractLookupDict,
    DelegationSyncMixin,
    early_late_mapping,
)
from lymph.types import DiagnoseType, PatternType

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

    logger.debug(f"Created sync callback for properties {names} of {this.get_name} edge.")
    return sync

# this here could probably be used to sync the edges for the different bilateral classes if we want to keep on using it
def init_edge_sync(
    property_names: list[str],
    this_edge_list: list[graph.Edge],
    other_edge_list: list[graph.Edge],
) -> None:
    """Initialize the callbacks to sync properties btw. Edges.

    Implementing this as a separate method allows a user in theory to initialize
    an arbitrary kind of symmetry between the two sides of the neck.
    """
    this_edge_names = [e.get_name for e in this_edge_list]
    other_edge_names = [e.get_name for e in other_edge_list]

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


class Midline(DelegationSyncMixin):
    """Models metastatic progression bilaterally with tumor lateralization.

    Model a bilateral lymphatic system where an additional risk factor can
    be provided in the data: Whether or not the primary tumor extended over the
    mid-sagittal line, or is located on the mid-saggital line.

    It is reasonable to assume (and supported by data) that an extension of the primary
    tumor significantly increases the risk for metastatic spread to the contralateral
    side of the neck. This class attempts to capture this using a simple
    assumption: We assume that the probability of spread to the contralateral
    side for patients *with* midline extension is larger than for patients
    *without* it, but smaller than the probability of spread to the ipsilateral
    side. Formally:

    .. math::
        b_c^{\\in} = \\alpha \\cdot b_i + (1 - \\alpha) \\cdot b_c^{\\not\\in}

    where :math:`b_c^{\\in}` is the probability of spread from the primary tumor
    to the contralateral side for patients with midline extension, and
    :math:`b_c^{\\not\\in}` for patients without. :math:`\\alpha` is the linear
    mixing parameter.
    """
    def __init__(
        self,
        graph_dict: dict[tuple[str], list[str]],
        use_mixing: bool = True,
        modalities_symmetric: bool = True,
        trans_symmetric: bool = True,
        unilateral_kwargs: dict[str, Any] | None = None,
        central_enabled: bool = True,
        **_kwargs
    ):
        """Initialize the model.

        The class is constructed in a similar fashion to the
        :py:class:`~lymph.models.Bilateral`: That class contains one
        :py:class:`~lymph.models.Unilateral` for each side of the neck, while this
        class will contain several instances of :py:class:`~lymph.models.Bilateral`,
        one for the ipsilateral side and two to three for the the contralateral side
        covering the cases a) no midline extension, b) midline extension, and c)
        central tumor location.

        Args:
            graph: Dictionary of the same kind as for initialization of
                :class:`System`. This graph will be passed to the constructors of
                two :class:`System` attributes of this class.
            use_mixing: Describe the contralateral base spread probabilities for the
                case of a midline extension as a linear combination between the base
                spread probs of the ipsilateral side and the ones of the contralateral
                side when no midline extension is present.
            trans_symmetric: If ``True``, the spread probabilities among the
                LNLs will be set symmetrically.
            central_enabled: If ``True``, a third bilateral class is produced
            which holds a model for patients with central tumor locations.

        The ``unilateral_kwargs`` are passed to all bilateral models.
        See Also:
            :class:`Bilateral`: Two of these are held as attributes by this
            class. One for the case of a mid-sagittal extension of the primary
            tumor and one for the case of no such extension.
        """
        super().__init__()
        self.central_enabled = central_enabled
        self.ext   = models.Bilateral(graph_dict= graph_dict,unilateral_kwargs=unilateral_kwargs, is_symmetric={'tumor_spread':False, "modalities": modalities_symmetric, "lnl_spread":trans_symmetric})
        self.noext = models.Bilateral(graph_dict= graph_dict,unilateral_kwargs=unilateral_kwargs, is_symmetric={'tumor_spread':False, "modalities": modalities_symmetric, "lnl_spread":trans_symmetric})
        if self.central_enabled:
            self.central = models.Bilateral(graph_dict= graph_dict,unilateral_kwargs=unilateral_kwargs, is_symmetric={'tumor_spread':True, "modalities": modalities_symmetric, "lnl_spread":trans_symmetric})

        self.use_mixing = use_mixing
        self.diag_time_dists = {}
        if self.use_mixing:
            self.alpha_mix = 0.

        self.modalities_symmetric = modalities_symmetric
        property_names = ["spread_prob"]
        if self.ext.ipsi.graph.is_trinary:
            property_names.append("micro_mod")
        delegated_attrs = [
            "max_time", "t_stages",
            "is_binary", "is_trinary",
        ]

        init_dict_sync(
            this=self.ext.ipsi.diag_time_dists,
            other=self.noext.ipsi.diag_time_dists,
        )
        if central_enabled:
            init_dict_sync(
                this=self.noext.ipsi.diag_time_dists,
                other=self.central.ipsi.diag_time_dists
                )

        if self.modalities_symmetric:
            delegated_attrs.append("modalities")
            init_dict_sync(
                this=self.ext.modalities,
                other=self.noext.modalities,
            )
            if central_enabled:
                init_dict_sync(
                    this=self.noext.modalities,
                    other=self.central.modalities,
                )
        self.init_synchronization()
        self.init_delegation(ext=delegated_attrs)

    def init_synchronization(self) -> None:
        """Initialize the synchronization of edges, modalities, and diagnose times."""
        # Sync spread probabilities
        property_names = ["spread_prob", "micro_mod"] if self.noext.ipsi.is_trinary else ["spread_prob"]
        noext_ipsi_tumor_edges = list(self.noext.ipsi.graph.tumor_edges.values())
        noext_ipsi_lnl_edges = list(self.noext.ipsi.graph.lnl_edges.values())
        noext_ipsi_edges = (
            noext_ipsi_tumor_edges + noext_ipsi_lnl_edges
        )
        ext_ipsi_tumor_edges = list(self.ext.ipsi.graph.tumor_edges.values())
        ext_ipsi_lnl_edges = list(self.ext.ipsi.graph.lnl_edges.values())
        ext_ipsi_edges = (
            ext_ipsi_tumor_edges
            + ext_ipsi_lnl_edges
        )


        init_edge_sync(
            property_names=property_names,
            this_edge_list=noext_ipsi_edges,
            other_edge_list=ext_ipsi_edges,
        )

        #The syncing below does not work properly. The ipsilateral central side is synced, but the contralateral central side is not synced. It seems like no callback is initiated when syncing in this manner

        # if self.central_enabled:
        #     central_ipsi_tumor_edges = list(self.central.ipsi.graph.tumor_edges.values())
        #     central_ipsi_lnl_edges = list(self.central.ipsi.graph.lnl_edges.values())
        #     central_ipsi_edges = (
        #         central_ipsi_tumor_edges
        #         + central_ipsi_lnl_edges
        #     )
        #     init_edge_sync(
        #         property_names=property_names,W
        #         this_edge_list=noext_ipsi_edges,
        #         other_edge_list=central_ipsi_edges,
        #     )

    def get_params(
        self):
        """Return the parameters of the model.
        Parameters are only returned as dictionary.
        """

        if self.use_mixing:
            return {'ipsi': self.noext.ipsi.get_params(as_dict=True),
                'no extension contra':self.noext.contra.get_params(as_dict=True),
                'mixing':self.alpha_mix}
        else:
            return {
                'ipsi':self.ext.ipsi.get_params(as_dict=True),
                'extension contra':self.ext.contra.get_params(as_dict=True),
                'no extension contra':self.noext.contra.get_params(as_dict=True)}


    def assign_params(
        self,
        *new_params_args,
        **new_params_kwargs,
    ) -> tuple[Iterator[float, dict[str, float]]]:
        """Assign new parameters to the model.

        This works almost exactly as the bilateral model's
        :py:meth:`~lymph.models.Bilateral.assign_params` method. However the assignment of parametrs
        with an array is disabled as it gets to messy with such a large parameter space.
        For universal parameters, the prefix is not needed as they are directly
        sent to the noextension ipsilateral side, which then triggers a sync callback.
        """
        if self.use_mixing:
            extension_kwargs = {}
            no_extension_kwargs = {}
            central_kwargs = {}
            for key, value in new_params_kwargs.items():
                if 'mixing' in key:
                    self.alpha_mix = value
                else:
                    no_extension_kwargs[key] = value
            remaining_args, remainings_kwargs = self.noext.set_params(*new_params_args, **no_extension_kwargs)
            for key in no_extension_kwargs.keys():
                if 'contra_primary' in key:
                    extension_kwargs[key] = self.alpha_mix * extension_kwargs[(key.replace("contra", "ipsi"))] + (1. - self.alpha_mix) * no_extension_kwargs[key]
                else:
                    extension_kwargs[key] = no_extension_kwargs[key]
            remaining_args, remainings_kwargs = self.ext.set_params(*remaining_args, **extension_kwargs)
            # If the syncing of the edges works properly, this below can be deleted.
            if self.central_enabled:
                for key in no_extension_kwargs.keys():
                    if 'contra' not in key:
                        central_kwargs[(key.replace("ipsi_", ""))] = no_extension_kwargs[key]
                remaining_args, remainings_kwargs = self.central.set_params(*new_params_args, **central_kwargs)
        else:
            ipsi_kwargs, noext_contra_kwargs, ext_contra_kwargs, general_kwargs, central_kwargs = {}, {}, {}, {}, {}

            for key, value in new_params_kwargs.items():
                if "ipsi_" in key:
                    ipsi_kwargs[key.replace("ipsi_", "")] = value
                elif "noext" in key:
                    noext_contra_kwargs[key.replace("contra_noext_", "")] = value
                elif 'ext' in key:
                    ext_contra_kwargs[key.replace("contra_ext_", "")] = value
                else:
                    if 'contra' in key:
                        warnings.warn(
                "'contra' keys were assigned without 'ext' or 'noext' defined. For a non-mixture model"
                "For a non mixture model these values have no meaning.")
                    else:
                        general_kwargs[key] = value

            remaining_args, remainings_kwargs = self.ext.ipsi.set_params(
                *new_params_args, **ipsi_kwargs, **general_kwargs
            )
            remaining_args, remainings_kwargs = self.noext.contra.set_params(
                *remaining_args, **noext_contra_kwargs, **remainings_kwargs, **general_kwargs
            )
            remaining_args, remainings_kwargs = self.ext.contra.set_params(
                *remaining_args, **ext_contra_kwargs, **remainings_kwargs, **general_kwargs
            )
            if self.central_enabled:
                for key in ipsi_kwargs.keys():
                        central_kwargs[(key.replace("ipsi_", ""))] = ipsi_kwargs[key]
                print(ipsi_kwargs)
                print(general_kwargs)
                remaining_args, remainings_kwargs = self.central.set_params(*new_params_args, **central_kwargs, **general_kwargs)

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
        return self.ext.modalities

    @modalities.setter
    def modalities(self, new_modalities) -> None:
        """Set the diagnostic modalities of the model."""
        if not self.modalities_symmetric:
            raise AttributeError(
                "The modalities are not symmetric. Please set them via the "
                "`ipsi` or `contra` attributes."
            )
        self.ext.modalities = new_modalities


    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        mapping: callable = early_late_mapping,
    ) -> None:
        """Load patient data into the model.

        This amounts to calling the :py:meth:`~lymph.models.Unilateral.load_patient_data`
        method on both models.
        """
        if self.central_enabled:
            ext_data = patient_data.loc[(patient_data[("tumor", "1", "extension")] == True) & (patient_data[("tumor", "1", "central")] != True)]
            noext_data = patient_data.loc[~patient_data[("tumor", "1", "extension")]]
            central = patient_data[patient_data[("tumor", "1", "central")].notna() & patient_data[("tumor", "1", "central")]]
            self.central.load_patient_data(central, mapping)
        else:
            ext_data = patient_data.loc[(patient_data[("tumor", "1", "extension")] == True)]
            noext_data = patient_data.loc[~patient_data[("tumor", "1", "extension")]]
        self.ext.load_patient_data(ext_data, mapping)
        self.noext.load_patient_data(noext_data, mapping)


    def likelihood(
        self,
        data: OPTIONAL[pd.DataFrame] = None,
        given_param_kwargs: dict[str, float] | None = None,
        log: bool = True,
        mode: str = 'HMM'
    ) -> float:
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

            :meth:`Unilateral.likelihood`: The log-likelihood function of
            the unilateral system.

            :meth:`Bilateral.likelihood`: The (log-)likelihood function of the
            bilateral system.
        """
        if data is not None:
            self.patient_data = data

        if given_param_kwargs is None:
            given_param_kwargs = {}

        try:
            self.assign_params(**given_param_kwargs)
        except ValueError:
            return -np.inf if log else 0.

        llh = 0. if log else 1.
        if log:
            llh += self.ext.likelihood(log = log, mode = mode)
            llh += self.noext.likelihood(log = log, mode = mode)
            if self.central_enabled:
                llh += self.central.likelihood(log = log, mode = mode)
        else:
            llh *= self.ext.likelihood(log = log, mode = mode)
            llh *= self.noext.likelihood(log = log, mode = mode)
            if self.central_enabled:
                llh *= self.central.likelihood(log = log, mode = mode)

        return llh


    def risk(
        self,
        involvement: PatternType | None = None,
        given_param_args: Iterable[float] | None = None,
        given_param_kwargs: dict[str, float] | None = None,
        given_diagnoses: dict[str, DiagnoseType] | None = None,
        t_stage: str = "early",
        midline_extension: bool = False,
        central: bool = False,
        mode: str = "HMM",
    ) -> float:
        """Compute the risk of nodal involvement given a specific diagnose.

        Args:
            spread_probs: Set ot new spread parameters. This also contains the
                mixing parameter alpha in the last position.
            midline_extension: Whether or not the patient's tumor extends over
                the mid-sagittal line.

        See Also:
            :meth:`Bilateral.risk`: Depending on whether or not the patient's
            tumor does extend over the midline, the risk function of the
            respective :class:`Bilateral` instance gets called.
        """
        if given_param_args is not None:
            self.assign_params(*given_param_args)
        if given_param_kwargs is not None:
            self.assign_params(**given_param_kwargs)
        if central:
            return self.central.risk(given_diagnoses = given_diagnoses,t_stage = t_stage, involvement = involvement)
        if midline_extension:
            return self.ext.risk(given_diagnoses = given_diagnoses,t_stage = t_stage, involvement = involvement)
        return self.noext.risk(given_diagnoses = given_diagnoses,t_stage = t_stage, involvement = involvement)



    # def generate_dataset(
    #     self,
    #     num_patients: int,
    #     stage_dist: dict[str, float],
    # ) -> pd.DataFrame:
    #     """Generate/sample a pandas :class:`DataFrame` from the defined network.

    #     Args:
    #         num_patients: Number of patients to generate.
    #         stage_dist: Probability to find a patient in a certain T-stage.
    #     """
    #     # TODO: check if this still works
    #     drawn_t_stages, drawn_diag_times = self.diag_time_dists.draw(
    #         dist=stage_dist, size=num_patients
    #     )

    #     drawn_obs_ipsi = self.ipsi._draw_patient_diagnoses(drawn_diag_times)
    #     drawn_obs_contra = self.contra._draw_patient_diagnoses(drawn_diag_times)
    #     drawn_obs = np.concatenate([drawn_obs_ipsi, drawn_obs_contra], axis=1)

    #     # construct MultiIndex for dataset from stored modalities
    #     sides = ["ipsi", "contra"]
    #     modalities = list(self.modalities.keys())
    #     lnl_names = [lnl.name for lnl in self.ipsi.graph._lnls]
    #     multi_cols = pd.MultiIndex.from_product([sides, modalities, lnl_names])

    #     # create DataFrame
    #     dataset = pd.DataFrame(drawn_obs, columns=multi_cols)
    #     dataset = dataset.reorder_levels(order=[1, 0, 2], axis="columns")
    #     dataset = dataset.sort_index(axis="columns", level=0)
    #     dataset[('info', 'tumor', 't_stage')] = drawn_t_stages

    #     return dataset
