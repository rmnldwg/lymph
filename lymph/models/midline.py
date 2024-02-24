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
        use_midext_evo: bool = True,
        unilateral_kwargs: dict[str, Any] | None = None,
        **_kwargs
    ):
        """Initialize the model.

        The class is constructed in a similar fashion to the :py:class:`~.Bilateral`:
        That class contains one :py:class:`~.Unilateral` for each side of the neck,
        while this class will contain several instances of :py:class:`~.Bilateral`,
        one for the ipsilateral side and two to three for the the contralateral side
        covering the cases a) no midline extension, b) midline extension, and c)
        central tumor location.

        Added keyword arguments in this constructor are ``use_mixing``, which controls
        whether to use the above described mixture of spread parameters from tumor to
        the LNLs. And ``use_central``, which controls whether to use a third
        :py:class:`~.Bilateral` model for the case of a central tumor location.

        The parameter ``use_midext_evo`` decides whether the tumor's midline extions
        should be considered a random variable, in which case it is evolved like the
        state of the LNLs, or not.

        The ``unilateral_kwargs`` are passed to all bilateral models.

        See Also:
            :py:class:`Bilateral`: Two of these are held as attributes by this
            class. One for the case of a mid-sagittal extension of the primary
            tumor and one for the case of no such extension.
        """
        if is_symmetric is None:
            is_symmetric = {}

        is_symmetric["tumor_spread"] = is_symmetric.get("tumor_spread", False)
        is_symmetric["lnl_spread"] = is_symmetric.get("lnl_spread", True)

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

        self.use_midext_evo = use_midext_evo
        if self.use_midext_evo and use_central:
            raise ValueError(
                "Evolution to central tumor not yet implemented. Choose to use either "
                "the central model or the midline extension evolution."
                # Actually, this shouldn't be too hard, but we still need to think
                # about it for a bit.
            )
        central_child = {}
        if use_central:
            self._central = models.Bilateral(
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

        self.midext_prob = 0.

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


    @classmethod
    def trinary(cls, *args, **kwargs) -> Midline:
        """Create a trinary model."""
        unilateral_kwargs = kwargs.pop("unilateral_kwargs", {})
        unilateral_kwargs["allowed_states"] = [0, 1, 2]
        return cls(*args, unilateral_kwargs=unilateral_kwargs, **kwargs)


    @property
    def is_trinary(self) -> bool:
        """Return whether the model is trinary."""
        if self.ext.is_trinary != self.noext.is_trinary:
            raise ValueError("The bilateral models must have the same trinary status.")

        if self.use_central and self.central.is_trinary != self.ext.is_trinary:
            raise ValueError("The bilateral models must have the same trinary status.")

        return self.ext.is_trinary


    @property
    def midext_prob(self) -> float:
        """Return the probability of midline extension."""
        if hasattr(self, "_midext_prob"):
            return self._midext_prob
        return 0.

    @midext_prob.setter
    def midext_prob(self, value: float) -> None:
        """Set the probability of midline extension."""
        if value is not None and not 0. <= value <= 1.:
            raise ValueError("The midline extension prob must be in the range [0, 1].")
        self._midext_prob = value


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
        return hasattr(self, "_central")

    @property
    def central(self) -> models.Bilateral:
        """Return the central model."""
        return self._central


    def get_tumor_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> dict[str, float] | Iterable[float]:
        """Return the tumor spread parameters of the model.

        If the model uses the mixing parameter, the returned params will contain the
        ipsilateral spread from tumor to LNLs, the contralateral ones for the case of
        no midline extension, and the mixing parameter. Otherwise, it will contain the
        contralateral params for the cases of present and absent midline extension.
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

        if as_flat or not as_dict:
            params = flatten(params)

        return params if as_dict else params.values()


    def get_lnl_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> dict[str, float] | Iterable[float]:
        """Return the LNL spread parameters of the model.

        Depending on the value of ``is_symmetric["lnl_spread"]``, the returned params
        may contain only one set of spread parameters (if ``True``) or one for the ipsi-
        and one for the contralateral side (if ``False``).
        """
        ext_lnl_params = self.ext.get_lnl_spread_params(as_flat=False)
        noext_lnl_params = self.noext.get_lnl_spread_params(as_flat=False)

        if ext_lnl_params != noext_lnl_params:
            raise ValueError(
                "LNL spread params not synched between ext and noext models. "
                "Returning the ext params."
            )

        if self.use_central:
            central_lnl_params = self.central.get_lnl_spread_params(as_flat=False)
            if central_lnl_params != ext_lnl_params:
                warnings.warn(
                    "LNL spread params not synched between central and ext models. "
                    "Returning the ext params."
                )

        if as_flat or not as_dict:
            ext_lnl_params = flatten(ext_lnl_params)

        return ext_lnl_params if as_dict else ext_lnl_params.values()


    def get_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> dict[str, float] | Iterable[float]:
        """Return the spread parameters of the model.

        This combines the returned values from the calls to
        :py:meth:`get_tumor_spread_params` and :py:meth:`get_lnl_spread_params`.
        """
        params = self.get_tumor_spread_params(as_flat=False)
        lnl_spread_params = self.get_lnl_spread_params(as_flat=False)

        if self.is_symmetric["lnl_spread"]:
            params.update(lnl_spread_params)
        else:
            if "contra" not in params:
                params["contra"] = {}
            params["ipsi"].update(lnl_spread_params["ipsi"])
            params["contra"].update(lnl_spread_params["contra"])

        if as_flat or not as_dict:
            params = flatten(params)

        return params if as_dict else params.values()


    def get_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> Iterable[float] | dict[str, float]:
        """Return all the parameters of the model.

        This includes the spread parameters from the call to :py:meth:`get_spread_params`
        and the distribution parameters from the call to :py:meth:`get_distribution_params`.
        """
        params = self.get_spread_params(as_flat=as_flat)
        params.update(self.get_distribution_params(as_flat=as_flat))
        params["mixing"] = self.mixing_param
        params["midext_prob"] = self.midext_prob

        if as_flat or not as_dict:
            params = flatten(params)

        return params if as_dict else params.values()


    def set_tumor_spread_params(
        self, *args: float, **kwargs: float,
    ) -> Iterable[float] | dict[str, float]:
        """Set the spread parameters of the midline model.

        In analogy to the :py:meth:`get_tumor_spread_params` method, this method sets
        the parameters describing how the tumor spreads to the LNLs. How many params
        to provide to this model depends on the value of the ``use_mixing`` and the
        ``use_central`` attributes. Have a look at what the
        :py:meth:`get_tumor_spread_params` method returns for an insight in what you
        can provide.
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

        return args


    def set_lnl_spread_params(self, *args: float, **kwargs: float) -> Iterable[float]:
        """Set the LNL spread parameters of the midline model.

        This works exactly like the :py:meth:`.Bilateral.set_lnl_spread_params` for the
        user, but under the hood, the parameters also need to be distributed to two or
        three instances of :py:class:`~.Bilateral` depending on the value of the
        ``use_central`` attribute.
        """
        kwargs, global_kwargs = unflatten_and_split(
            kwargs, expected_keys=["ipsi", "noext", "ext", "contra"],
        )
        ipsi_kwargs = global_kwargs.copy()
        ipsi_kwargs.update(kwargs.get("ipsi", {}))

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


    def set_spread_params(self, *args: float, **kwargs: float) -> Iterable[float]:
        """Set the spread parameters of the midline model."""
        args = self.set_tumor_spread_params(*args, **kwargs)
        return self.set_lnl_spread_params(*args, **kwargs)


    def set_params(
        self, *args: float, **kwargs: float,
    ) -> Iterable[float] | dict[str, float]:
        """Set all parameters of the model.

        Combines the calls to :py:meth:`set_spread_params` and
        :py:meth:`set_distribution_params`.
        """
        args = self.set_spread_params(*args, **kwargs)
        first, args = popfirst(args)
        self.mixing_param = kwargs.get("mixing", first) or self.mixing_param
        first, args = popfirst(args)
        self.midext_prob = kwargs.get("midext_prob", first) or self.midext_prob
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


    def comp_contra_dist_evolution(self) -> tuple[np.ndarray, np.ndarray]:
        """Evolve contra side as mixture of with & without midline extension."""
        noext_contra_dist_evo = np.zeros(
            shape=(self.max_time + 1, len(self.noext.contra.state_list))
        )
        noext_contra_dist_evo[0,0] = 1.

        ext_contra_dist_evo = np.zeros(
            shape=(self.max_time + 1, len(self.ext.contra.state_list))
        )
        if not self.use_midext_evo:
            noext_contra_dist_evo[0,0] = (1. - self.midext_prob)
            ext_contra_dist_evo[0,0] = self.midext_prob

        for t in range(self.max_time):
            # When evolving over the midline extension state, there's a chance at any
            # time step that the tumor grows over the midline and starts spreading to
            # the contralateral side more aggressively.
            if self.use_midext_evo:
                noext_contra_dist_evo[t+1] = (
                    (1. - self.midext_prob) * noext_contra_dist_evo[t]
                ) @ self.noext.contra.transition_matrix
                ext_contra_dist_evo[t+1] = (
                    self.midext_prob * noext_contra_dist_evo[t]
                    + ext_contra_dist_evo[t]
                ) @ self.ext.contra.transition_matrix

            # When we do not evolve, the tumor is considered lateralized or extending
            # over the midline from the start.
            else:
                noext_contra_dist_evo[t+1] = (
                    noext_contra_dist_evo[t] @ self.noext.contra.transition_matrix
                )
                ext_contra_dist_evo[t+1] = (
                    ext_contra_dist_evo[t] @ self.ext.contra.transition_matrix
                )

        return noext_contra_dist_evo, ext_contra_dist_evo


    def likelihood(
        self,
        given_params: Iterable[float] | dict[str, float] | None = None,
        log: bool = True,
        for_t_stage: str | None = None,
    ) -> float:
        """Compute the (log-)likelihood of the stored data given the model (and params).

        See the documentation of :py:meth:`lymph.types.Model.likelihood` for more
        information on how to use the ``given_params`` parameter.

        Returns the log-likelihood if ``log`` is set to ``True``. Note that in contrast
        to the :py:class:`~.Bilateral` model, the midline model does not support the
        Bayesian network mode.

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

        llh = 0. if log else 1.

        ipsi_dist_evo = self.ext.ipsi.comp_dist_evolution()
        contra_dist_evo = {}
        contra_dist_evo["ext"], contra_dist_evo["noext"] = self.comp_contra_dist_evolution()

        t_stages = self.t_stages if for_t_stage is None else [for_t_stage]
        for stage in t_stages:
            diag_time_matrix = np.diag(self.get_distribution(stage).pmf)
            # see the `Bilateral` model for why this is done in this way.
            for case in ["ext", "noext"]:
                joint_state_dist = (
                    ipsi_dist_evo.T
                    @ diag_time_matrix
                    @ contra_dist_evo[case]
                )
                joint_diagnose_dist = np.sum(
                    getattr(self, case).ipsi.diagnose_matrices[stage]
                    * (
                        joint_state_dist
                        @ getattr(self, case).contra.diagnose_matrices[stage]
                    )
                )
                if log:
                    llh += np.sum(np.log(joint_diagnose_dist))
                else:
                    llh *= np.prod(joint_diagnose_dist)

        if self.use_central:
            if log:
                llh += self.central.likelihood(log=log, for_t_stage=for_t_stage)
            else:
                llh *= self.central.likelihood(log=log, for_t_stage=for_t_stage)

        return llh


    def risk(
        self,
        involvement: PatternType | None = None,
        given_params: Iterable[float] | dict[str, float] | None = None,
        given_diagnoses: dict[str, DiagnoseType] | None = None,
        t_stage: str = "early",
        midline_extension: bool = False,
        central: bool = False,
        mode: str = "HMM",
    ) -> float:
        """Compute the risk of nodal involvement ``given_diagnoses``.

        In addition to the arguments of the :py:meth:`.Bilateral.risk` method, this
        also allows specifying if the patient's tumor extended over the mid-sagittal
        line (``midline_extension=True``) or if it was even located right on that line
        (``central=True``).

        For logical reasons, ``midline_extension=False`` makes no sense if
        ``central=True`` and is thus ignored.
        """
        if given_params is None:
            pass
        elif isinstance(given_params, dict):
            self.set_params(**given_params)
        else:
            self.set_params(*given_params)

        if central:
            return self.central.risk(
                given_diagnoses=given_diagnoses,
                t_stage=t_stage,
                involvement=involvement,
                mode=mode,
            )
        if midline_extension:
            return self.ext.risk(
                given_diagnoses=given_diagnoses,
                t_stage=t_stage,
                involvement=involvement,
                mode=mode,
            )
        return self.noext.risk(
            given_diagnoses=given_diagnoses,
            t_stage=t_stage,
            involvement=involvement,
            mode=mode,
        )
