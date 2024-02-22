from __future__ import annotations

import logging
import warnings
from typing import Any, Iterable

import numpy as np
import pandas as pd

from lymph import diagnose_times, modalities, models, types
from lymph.helper import (
    early_late_mapping,
    flatten,
    popfirst,
    unflatten_and_split,
)
from lymph.types import DiagnoseType, PatternType

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)


EXT_COL = ("tumor", "1", "extension")
CENTRAL_COL = ("tumor", "1", "central")



class Midline(
    diagnose_times.Composite,
    modalities.Composite,
    types.Model,
):
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
        is_symmetric: dict[str, bool] | None = None,
        use_mixing: bool = True,
        use_central: bool = True,
        unilateral_kwargs: dict[str, Any] | None = None,
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
        if is_symmetric is None:
            is_symmetric = {
                "tumor_spread": False,
                "lnl_spread": True,
            }
        if is_symmetric["tumor_spread"]:
            raise ValueError(
                "If you want the tumor spread to be symmetric, consider using the "
                "Bilateral class."
            )
        self.is_symmetric = is_symmetric

        self.ext = models.Bilateral(
            graph_dict=graph_dict,
            unilateral_kwargs=unilateral_kwargs,
            is_symmetric=self.is_symmetric,
        )
        self.noext = models.Bilateral(
            graph_dict=graph_dict,
            unilateral_kwargs=unilateral_kwargs,
            is_symmetric=self.is_symmetric,
        )
        central_child = {}
        if use_central:
            self.central = models.Bilateral(
                graph_dict=graph_dict,
                unilateral_kwargs=unilateral_kwargs,
                is_symmetric={
                    "tumor_spread": True,
                    "lnl_spread": self.is_symmetric["lnl_spread"],
                },
            )
            central_child = {"central": self.central}

        if use_mixing:
            self.mixing_param = 0.

        diagnose_times.Composite.__init__(
            self,
            distribution_children={"ext": self.ext, "noext": self.noext, **central_child},
            is_distribution_leaf=False,
        )
        modalities.Composite.__init__(
            self,
            modality_children={"ext": self.ext, "noext": self.noext, **central_child},
            is_modality_leaf=False,
        )


    @property
    def is_trinary(self) -> bool:
        """Return whether the model is trinary."""
        if self.ext.is_trinary != self.noext.is_trinary:
            raise ValueError("The bilateral models must have the same trinary status.")

        if self.use_central and self.central.is_trinary != self.ext.is_trinary:
            raise ValueError("The bilateral models must have the same trinary status.")

        return self.ext.is_trinary


    @property
    def mixing_param(self) -> float | None:
        """Return the mixing parameter."""
        if hasattr(self, "_mixing_param"):
            return self._mixing_param

        return None

    @mixing_param.setter
    def mixing_param(self, value: float) -> None:
        """Set the mixing parameter."""
        if value is not None and not 0. <= value <= 1.:
            raise ValueError("The mixing parameter must be in the range [0, 1].")

        self._mixing_param = value

    @property
    def use_mixing(self) -> bool:
        """Return whether the model uses a mixing parameter."""
        return hasattr(self, "_mixing_param")

    @property
    def use_central(self) -> bool:
        """Return whether the model uses a central model."""
        return hasattr(self, "central")


    def get_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> dict[str, float] | Iterable[float]:
        """Return the spread parameters of the model.

        TODO: enrich docstring
        """
        params = {}
        params["ipsi"] = self.ext.ipsi.get_tumor_spread_params(as_flat=as_flat)

        if self.use_mixing:
            params["contra"] = self.noext.contra.get_tumor_spread_params(as_flat=as_flat)
            params["mixing"] = self.mixing_param
        else:
            params["noext"] = {
                "contra": self.noext.contra.get_tumor_spread_params(as_flat=as_flat)
            }
            params["ext"] = {
                "contra": self.ext.contra.get_tumor_spread_params(as_flat=as_flat)
            }

        if self.is_symmetric["lnl_spread"]:
            params.update(self.ext.ipsi.get_lnl_spread_params(as_flat=as_flat))
        else:
            if "contra" not in params:
                params["contra"] = {}
            params["ipsi"].update(self.ext.ipsi.get_lnl_spread_params(as_flat=as_flat))
            params["contra"].update(self.noext.contra.get_lnl_spread_params(as_flat=as_flat))

        if as_flat or not as_dict:
            params = flatten(params)

        return params if as_dict else params.values()


    def get_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> Iterable[float] | dict[str, float]:
        """Return the parameters of the model.

        TODO: enrich docstring
        """
        params = self.get_spread_params(as_flat=as_flat)
        params.update(self.get_distribution_params(as_flat=as_flat))

        if as_flat or not as_dict:
            params = flatten(params)

        return params if as_dict else params.values()


    def set_spread_params(
        self, *args: float, **kwargs: float,
    ) -> Iterable[float] | dict[str, float]:
        """Set the spread parameters of the midline model.

        TODO: enrich docstring
        """
        kwargs, global_kwargs = unflatten_and_split(
            kwargs, expected_keys=["ipsi", "noext", "ext", "contra"],
        )

        # first, take care of ipsilateral tumor spread (same for all models)
        ipsi_kwargs = global_kwargs.copy()
        ipsi_kwargs.update(kwargs.get("ipsi", {}))
        if self.use_central:
            self.central.set_spread_params(*args, **ipsi_kwargs)
        self.ext.ipsi.set_tumor_spread_params(*args, **ipsi_kwargs)
        args = self.noext.ipsi.set_tumor_spread_params(*args, **ipsi_kwargs)

        # then, take care of contralateral tumor spread
        if self.use_mixing:
            contra_kwargs = global_kwargs.copy()
            contra_kwargs.update(kwargs.get("contra", {}))
            args = self.noext.contra.set_tumor_spread_params(*args, **contra_kwargs)
            mixing_param, args = popfirst(args)
            mixing_param = global_kwargs.get("mixing", mixing_param) or self.mixing_param
            self.mixing_param = global_kwargs.get("mixing", mixing_param)

            ext_contra_kwargs = {}
            for (key, ipsi_param), noext_contra_param in zip(
                self.ext.ipsi.get_tumor_spread_params().items(),
                self.noext.contra.get_tumor_spread_params().values(),
            ):
                ext_contra_kwargs[key] = (
                    self.mixing_param * ipsi_param
                    + (1. - self.mixing_param) * noext_contra_param
                )
            self.ext.contra.set_tumor_spread_params(**ext_contra_kwargs)

        else:
            noext_contra_kwargs = global_kwargs.copy()
            noext_contra_kwargs.update(kwargs.get("noext", {}).get("contra", {}))
            args = self.noext.contra.set_tumor_spread_params(*args, **noext_contra_kwargs)

            ext_contra_kwargs = global_kwargs.copy()
            ext_contra_kwargs.update(kwargs.get("ext", {}).get("contra", {}))
            args = self.ext.contra.set_tumor_spread_params(*args, **ext_contra_kwargs)

        # finally, take care of LNL spread
        if self.is_symmetric["lnl_spread"]:
            if self.use_central:
                self.central.ipsi.set_lnl_spread_params(*args, **global_kwargs)
                self.central.contra.set_lnl_spread_params(*args, **global_kwargs)
            self.ext.ipsi.set_lnl_spread_params(*args, **global_kwargs)
            self.ext.contra.set_lnl_spread_params(*args, **global_kwargs)
            self.noext.ipsi.set_lnl_spread_params(*args, **global_kwargs)
            args = self.noext.contra.set_lnl_spread_params(*args, **global_kwargs)

        else:
            if self.use_central:
                self.central.ipsi.set_lnl_spread_params(*args, **ipsi_kwargs)
            self.ext.ipsi.set_lnl_spread_params(*args, **ipsi_kwargs)
            args = self.noext.ipsi.set_lnl_spread_params(*args, **ipsi_kwargs)

            contra_kwargs = global_kwargs.copy()
            contra_kwargs.update(kwargs.get("contra", {}))
            if self.use_central:
                self.central.contra.set_lnl_spread_params(*args, **contra_kwargs)
            self.ext.contra.set_lnl_spread_params(*args, **contra_kwargs)
            args = self.noext.contra.set_lnl_spread_params(*args, **contra_kwargs)

        return args


    def set_params(
        self, *args: float, **kwargs: float,
    ) -> Iterable[float] | dict[str, float]:
        """Assign new parameters to the model.

        TODO: enrich docstring
        """
        args = self.set_spread_params(*args, **kwargs)
        return self.set_distribution_params(*args, **kwargs)


    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        mapping: callable = early_late_mapping,
    ) -> None:
        """Load patient data into the model.

        This amounts to sorting the patients into three bins:
        1. Patients whose tumor is clearly laterlaized, meaning the column
            ``("tumor", "1", "extension")`` reports ``False``. These get assigned to
            the :py:attr:`noext` attribute.
        2. Those with a central tumor, indicated by ``True`` in the column
            ``("tumor", "1", "central")``. If the :py:attr:`use_central` attribute is
            set to ``True``, these patients are assigned to the :py:attr:`central`
            model. Otherwise, they are assigned to the :py:attr:`ext` model.
        3. The rest, which amounts to patients whose tumor extends over the mid-sagittal
            line but is not central, i.e., symmetric w.r.t to the mid-sagittal line.
            These are assigned to the :py:attr:`ext` model.

        The split data is sent to the :py:meth:`lymph.models.Bilateral.load_patient_data`
        method of the respective models.
        """
        # pylint: disable=singleton-comparison
        is_lateralized = patient_data[EXT_COL] == False
        self.noext.load_patient_data(patient_data[is_lateralized], mapping)

        if self.use_central:
            is_central = patient_data[CENTRAL_COL] == True
            self.central.load_patient_data(patient_data[is_central], mapping)
            self.ext.load_patient_data(patient_data[~is_lateralized & ~is_central], mapping)

        else:
            self.ext.load_patient_data(patient_data[~is_lateralized], mapping)


    def likelihood(
        self,
        given_params: Iterable[float] | dict[str, float] | None = None,
        log: bool = True,
        mode: str = "HMM",
        for_t_stage: str | None = None,
    ) -> float:
        """Compute the (log-)likelihood of the stored data given the model (and params).

        See the documentation of :py:meth:`lymph.types.Model.likelihood` for more
        information on how to use the ``given_params`` parameter.

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
        try:
            # all functions and methods called here should raise a ValueError if the
            # given parameters are invalid...
            if given_params is None:
                pass
            elif isinstance(given_params, dict):
                self.set_params(**given_params)
            else:
                self.set_params(*given_params)
        except ValueError:
            return -np.inf if log else 0.

        kwargs = {"log": log, "mode": mode, "for_t_stage": for_t_stage}
        llh = 0. if log else 1.
        if log:
            llh += self.ext.likelihood(**kwargs)
            llh += self.noext.likelihood(**kwargs)
            if self.use_central:
                llh += self.central.likelihood(**kwargs)
        else:
            llh *= self.ext.likelihood(**kwargs)
            llh *= self.noext.likelihood(**kwargs)
            if self.use_central:
                llh *= self.central.likelihood(**kwargs)

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
            self.set_params(*given_param_args)
        if given_param_kwargs is not None:
            self.set_params(**given_param_kwargs)
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
