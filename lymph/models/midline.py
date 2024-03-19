from __future__ import annotations

import logging
import warnings
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from lymph import diagnose_times, matrix, modalities, models, types, utils

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
        graph_dict: types.GraphDictType,
        is_symmetric: dict[str, bool] | None = None,
        use_mixing: bool = True,
        use_central: bool = True,
        use_midext_evo: bool = True,
        marginalize_unknown: bool = True,
        uni_kwargs: dict[str, Any] | None = None,
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

        With ``marginalize_unknown`` (default: ``True``), the model will also load
        patients with unknown midline extension status into the model and marginalize
        over their state of midline extension when computing the likelihood. This extra
        data is stored in a :py:class:`~.Bilateral` instance accessible via the
        attribute ``"unknown"``. Note that this bilateral instance does not get updated
        parameters or any other kind of attention. It is solely used to store the data
        and generate diagnose matrices for those data.

        The ``uni_kwargs`` are passed to all bilateral models.

        See Also:
            :py:class:`Bilateral`: Two to four of these are held as attributes by this
            class. One for the case of a mid-sagittal extension of the primary
            tumor, one for the case of no such extension, (possibly) one for the case of
            a central/symmetric tumor, and (possibly) one for the case of unknown
            midline extension status.
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
            uni_kwargs=uni_kwargs,
            is_symmetric=self.is_symmetric,
        )
        self.noext = models.Bilateral(
            graph_dict=graph_dict,
            uni_kwargs=uni_kwargs,
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
        other_children = {}
        if use_central:
            self._central = models.Bilateral(
                graph_dict=graph_dict,
                uni_kwargs=uni_kwargs,
                is_symmetric={
                    "tumor_spread": True,
                    "lnl_spread": self.is_symmetric["lnl_spread"],
                },
            )
            other_children["central"] = self.central

        if marginalize_unknown:
            self._unknown = models.Bilateral(
                graph_dict=graph_dict,
                uni_kwargs=uni_kwargs,
                is_symmetric=self.is_symmetric,
            )
            other_children["unknown"] = self.unknown

        if use_mixing:
            self.mixing_param = 0.

        self.midext_prob = 0.

        diagnose_times.Composite.__init__(
            self,
            distribution_children={"ext": self.ext, "noext": self.noext, **other_children},
            is_distribution_leaf=False,
        )
        modalities.Composite.__init__(
            self,
            modality_children={"ext": self.ext, "noext": self.noext, **other_children},
            is_modality_leaf=False,
        )


    @classmethod
    def trinary(cls, *args, **kwargs) -> Midline:
        """Create a trinary model."""
        uni_kwargs = kwargs.pop("uni_kwargs", {})
        uni_kwargs["allowed_states"] = [0, 1, 2]
        return cls(*args, uni_kwargs=uni_kwargs, **kwargs)


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
        if self.use_central:
            return self._central
        raise AttributeError("This instance does not account for central tumors.")

    @property
    def marginalize_unknown(self) -> bool:
        """Return whether the model marginalizes over unknown midline extension."""
        return hasattr(self, "_unknown")

    @property
    def unknown(self) -> models.Bilateral:
        """Return the model storing the patients with unknown midline extension."""
        if self.marginalize_unknown:
            return self._unknown
        raise AttributeError(
            "This instance does not marginalize over unknown midline extension."
        )


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
            params = utils.flatten(params)

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
            ext_lnl_params = utils.flatten(ext_lnl_params)

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
            params = utils.flatten(params)

        return params if as_dict else params.values()


    def get_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return all the parameters of the model.

        This includes the spread parameters from the call to :py:meth:`get_spread_params`
        and the distribution parameters from the call to
        :py:meth:`~.diagnose_times.Composite.get_distribution_params`.
        """
        params = {}
        params["midext_prob"] = self.midext_prob
        params.update(self.get_spread_params(as_flat=as_flat))
        params.update(self.get_distribution_params(as_flat=as_flat))

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()


    def set_tumor_spread_params(
        self, *args: float, **kwargs: float,
    ) -> types.ParamsType:
        """Set the spread parameters of the midline model.

        In analogy to the :py:meth:`get_tumor_spread_params` method, this method sets
        the parameters describing how the tumor spreads to the LNLs. How many params
        to provide to this model depends on the value of the ``use_mixing`` and the
        ``use_central`` attributes. Have a look at what the
        :py:meth:`get_tumor_spread_params` method returns for an insight in what you
        can provide.
        """
        kwargs, global_kwargs = utils.unflatten_and_split(
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
            mixing_param, args = utils.popfirst(args)
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
        kwargs, global_kwargs = utils.unflatten_and_split(
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
    ) -> types.ParamsType:
        """Set all parameters of the model.

        Combines the calls to :py:meth:`.set_spread_params` and
        :py:meth:`.set_distribution_params`.
        """
        first, args = utils.popfirst(args)
        self.midext_prob = kwargs.get("midext_prob", first) or self.midext_prob
        args = self.set_spread_params(*args, **kwargs)
        return self.set_distribution_params(*args, **kwargs)


    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        mapping: callable = utils.early_late_mapping,
    ) -> None:
        """Load patient data into the model.

        This amounts to sorting the patients into three bins:

        1. Patients whose tumor is clearly laterlaized, meaning the column
           ``("tumor", "1", "extension")`` reports ``False``. These get assigned to
           the :py:attr:`.noext` attribute.
        2. Those with a central tumor, indicated by ``True`` in the column
           ``("tumor", "1", "central")``. If the :py:attr:`.use_central` attribute is
           set to ``True``, these patients are assigned to the :py:attr:`.central`
           model. Otherwise, they are assigned to the :py:attr:`.ext` model.
        3. The rest, which amounts to patients whose tumor extends over the mid-sagittal
           line but is not central, i.e., symmetric w.r.t to the mid-sagittal line.
           These are assigned to the :py:attr:`.ext` model.

        The split data is sent to the :py:meth:`.Bilateral.load_patient_data` method of
        the respective models.
        """
        # pylint: disable=singleton-comparison
        is_lateralized = patient_data[EXT_COL] == False
        has_extension = patient_data[EXT_COL] == True
        is_unknown = patient_data[EXT_COL].isna()
        self.noext.load_patient_data(patient_data[is_lateralized], mapping)

        if self.use_central:
            is_central = patient_data[CENTRAL_COL] == True
            has_extension = has_extension & ~is_central
            self.central.load_patient_data(patient_data[is_central], mapping)

        self.ext.load_patient_data(patient_data[has_extension], mapping)

        if self.marginalize_unknown and is_unknown.sum() > 0:
            self.unknown.load_patient_data(patient_data[is_unknown], mapping)
        elif is_unknown.sum() > 0:
            warnings.warn(
                f"Discarding {is_unknown.sum()} patients where midline extension "
                "is unknown."
            )


    def midext_evo(self) -> np.ndarray:
        """Evolve only the state of the midline extension."""
        midext_states = np.zeros(shape=(self.max_time + 1, 2), dtype=float)
        midext_states[0,0] = 1.

        midextransition_matrix = np.array([
            [1 - self.midext_prob, self.midext_prob],
            [0.                  , 1.              ],
        ])

        # compute involvement for all time steps
        for i in range(len(midext_states)-1):
            midext_states[i+1,:] = midext_states[i,:] @ midextransition_matrix
        return midext_states


    def contra_state_dist_evo(self) -> tuple[np.ndarray, np.ndarray]:
        """Evolve contra side as mixture of with & without midline extension."""
        noext_contra_dist_evo = self.noext.contra.state_dist_evo()
        ext_contra_dist_evo = self.ext.contra.state_dist_evo()

        if not self.use_midext_evo:
            noext_contra_dist_evo *= (1. - self.midext_prob)
            ext_contra_dist_evo *= self.midext_prob

        else:
            midext_evo = self.midext_evo()
            noext_contra_dist_evo *= midext_evo[:,0].reshape((-1, 1))
            ext_contra_dist_evo *= midext_evo[:,1].reshape((-1, 1))

        return noext_contra_dist_evo, ext_contra_dist_evo


    def state_dist(
        self,
        t_stage: str = "early",
        central: bool = False,
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Compute the joint over ipsi- & contralaleral hidden states and midline ext.

        If ``central=False``, the result has shape (2, num_states, num_states), where
        the first axis is for the midline extension status, the second for the
        ipsilateral state, and the third for the contralateral state.

        If ``central=True``, the result will be the state distribution of the central
        model's :py:meth:`.Bilateral.state_dist` method.
        """
        if central:
            return self.central.state_dist(t_stage, mode)

        ipsi_dist_evo = self.ext.ipsi.state_dist_evo()
        noext_contra_dist_evo, ext_contra_dist_evo = self.contra_state_dist_evo()

        if mode == "HMM":
            result = np.empty(shape=(2, ipsi_dist_evo.shape[1], ipsi_dist_evo.shape[1]))
            time_marg_matrix = np.diag(self.get_distribution(t_stage).pmf)
            result[0] = ipsi_dist_evo.T @ time_marg_matrix @ noext_contra_dist_evo
            result[1] = ipsi_dist_evo.T @ time_marg_matrix @ ext_contra_dist_evo
            return result

        raise NotImplementedError("Only HMM mode is supported as of now.")


    def _hmm_likelihood(self, log: bool = True, for_t_stage: str | None = None) -> float:
        """Compute the likelihood of the stored data under the hidden Markov model."""
        llh = 0. if log else 1.

        ipsi_dist_evo = self.ext.ipsi.state_dist_evo()
        contra_dist_evo = {}
        contra_dist_evo["noext"], contra_dist_evo["ext"] = self.contra_state_dist_evo()

        t_stages = self.t_stages if for_t_stage is None else [for_t_stage]
        for stage in t_stages:
            diag_time_matrix = np.diag(self.get_distribution(stage).pmf)
            num_states = ipsi_dist_evo.shape[1]
            marg_joint_state_dist = np.zeros(shape=(num_states, num_states))
            # see the `Bilateral` model for why this is done in this way.
            for case in ["ext", "noext"]:
                joint_state_dist = (
                    ipsi_dist_evo.T
                    @ diag_time_matrix
                    @ contra_dist_evo[case]
                )
                marg_joint_state_dist += joint_state_dist
                _model = getattr(self, case)
                patient_llhs = matrix.fast_trace(
                    _model.ipsi.diagnose_matrix(stage),
                    joint_state_dist @ _model.contra.diagnose_matrix(stage).T
                )
                llh = utils.add_or_mult(llh, patient_llhs, log=log)

            try:
                marg_patient_llhs = matrix.fast_trace(
                    self.unknown.ipsi.diagnose_matrix(stage),
                    marg_joint_state_dist @ self.unknown.contra.diagnose_matrix(stage).T
                )
                llh = utils.add_or_mult(llh, marg_patient_llhs, log=log)
            except AttributeError:
                # an AttributeError is raised both when the model has no `unknown`
                # attribute and when no data is loaded in the `unknown` model.
                pass

        if self.use_central:
            if log:
                llh += self.central.likelihood(log=log, for_t_stage=for_t_stage)
            else:
                llh *= self.central.likelihood(log=log, for_t_stage=for_t_stage)

        return llh


    def likelihood(
        self,
        given_params: types.ParamsType | None = None,
        log: bool = True,
        mode: Literal["HMM", "BN"] = "HMM",
        for_t_stage: str | None = None,
    ) -> float:
        """Compute the (log-)likelihood of the stored data given the model (and params).

        See the documentation of :py:meth:`.types.Model.likelihood` for more
        information on how to use the ``given_params`` parameter.

        Returns the log-likelihood if ``log`` is set to ``True``. Note that in contrast
        to the :py:class:`.Bilateral` model, the midline model does not support the
        Bayesian network mode.

        Note:
            The computation is faster if no parameters are given, since then the
            transition matrix does not need to be recomputed.

        See Also:
            :py:meth:`.Unilateral.likelihood`
                The corresponding unilateral function.
        """
        try:
            # all functions and methods called here should raise a ValueError if the
            # given parameters are invalid...
            utils.safe_set_params(self, given_params)
        except ValueError:
            return -np.inf if log else 0.

        if mode == "HMM":
            return self._hmm_likelihood(log, for_t_stage)

        raise NotImplementedError("Only HMM mode is supported as of now.")


    def risk(
        self,
        involvement: types.PatternType | None = None,
        given_params: types.ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnoses: dict[str, types.DiagnoseType] | None = None,
        t_stage: str = "early",
        midext: bool | None = None,
        central: bool = False,
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> float:
        """Compute the risk of nodal involvement ``given_diagnoses``.

        In addition to the arguments of the :py:meth:`.Bilateral.risk` method, this
        also allows specifying if the patient's tumor extended over the mid-sagittal
        line (``midext=True``) or if it was even located right on that line
        (``central=True``).

        For logical reasons, ``midext=False`` makes no sense if ``central=True`` and
        is thus ignored.

        Warning:
            As in the :py:meth:`.Bilateral.posterior_state_dist` method, you may
            provide a precomputed (joint) state distribution in the ``given_state_dist``
            argument. Here, this ``given_state_dist`` may be a 2D array, in which case
            it is assumed you know how it was computed and the arguments ``t_stage``,
            ``midext``, ``central``, and ``mode`` are ignored. If it is 3D, it should
            have the shape ``(2, num_states, num_states)`` and be the output of the
            :py:meth:`.Midline.state_dist` method. In this case, the ``midext``
            argument is *not* ignored: It may be used to select the correct state
            distribution (when ``True`` or ``False``), or marginalize over the midline
            extension status (when ``midext=None``).
        """
        # NOTE: When given a 2D state distribution, it does not matter which of the
        #       Bilateral models is used to compute the risk, since the state dist is
        #       is the only thing that could differ between models.
        if given_state_dist is None:
            utils.safe_set_params(self, given_params)
            given_state_dist = self.state_dist(t_stage, central, mode)

        if given_state_dist.ndim == 2:
            return self.ext.risk(
                involvement=involvement,
                given_state_dist=given_state_dist,
                given_diagnoses=given_diagnoses,
            )

        if central:
            raise ValueError("The `given_state_dist` must be 2D for the central model.")

        if midext is None:
            given_state_dist = np.sum(given_state_dist, axis=0)
        else:
            given_state_dist = given_state_dist[int(midext)]
            given_state_dist = given_state_dist / given_state_dist.sum()

        return self.ext.risk(
            involvement=involvement,
            given_state_dist=given_state_dist,
            given_diagnoses=given_diagnoses,
        )


    def draw_patients(
        self,
        num: int,
        stage_dist: Iterable[float],
        rng: np.random.Generator | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Draw ``num`` patients from the parameterized model."""
        if rng is None:
            rng = np.random.default_rng(seed)

        if sum(stage_dist) != 1.:
            warnings.warn("Sum of stage distribution is not 1. Renormalizing.")
            stage_dist = np.array(stage_dist) / sum(stage_dist)

        if self.use_central:
            raise NotImplementedError(
                "Drawing patients from the central model not yet supported."
            )

        drawn_t_stages = rng.choice(
            a=self.t_stages,
            p=stage_dist,
            size=num,
        )
        distributions = self.get_all_distributions()
        drawn_diag_times = np.array([
            distributions[t_stage].draw_diag_times(rng=rng)
            for t_stage in drawn_t_stages
        ])

        if self.use_midext_evo:
            midext_evo = self.midext_evo()
            drawn_midexts = np.array([
                rng.choice(a=[False, True], p=midext_evo[t])
                for t in drawn_diag_times
            ])
        else:
            drawn_midexts = rng.choice(
                a=[False, True],
                p=[1. - self.midext_prob, self.midext_prob],
                size=num,
            )

        ipsi_evo = self.ext.ipsi.state_dist_evo()
        drawn_diags = np.empty(shape=(num, len(self.ext.ipsi.obs_list)))
        for case in ["ext", "noext"]:
            case_model = getattr(self, case)
            drawn_ipsi_diags = utils.draw_diagnoses(
                diagnose_times=drawn_diag_times[drawn_midexts == (case == "ext")],
                state_evolution=ipsi_evo,
                observation_matrix=case_model.ipsi.observation_matrix(),
                possible_diagnoses=case_model.ipsi.obs_list,
                rng=rng,
                seed=seed,
            )
            drawn_contra_diags = utils.draw_diagnoses(
                diagnose_times=drawn_diag_times[drawn_midexts == (case == "ext")],
                state_evolution=case_model.contra.state_dist_evo(),
                observation_matrix=case_model.contra.observation_matrix(),
                possible_diagnoses=case_model.contra.obs_list,
                rng=rng,
                seed=seed,
            )
            drawn_case_diags = np.concatenate([drawn_ipsi_diags, drawn_contra_diags], axis=1)
            drawn_diags[drawn_midexts == (case == "ext")] = drawn_case_diags

        # construct MultiIndex with "ipsi" and "contra" at top level to allow
        # concatenation of the two separate drawn diagnoses
        sides = ["ipsi", "contra"]
        modality_names = list(self.get_all_modalities().keys())
        lnl_names = [lnl for lnl in self.ext.ipsi.graph.lnls.keys()]
        multi_cols = pd.MultiIndex.from_product([sides, modality_names, lnl_names])

        # reorder the column levels and thus also the individual columns to match the
        # LyProX format without mixing up the data
        dataset = pd.DataFrame(drawn_diags, columns=multi_cols)
        dataset = dataset.reorder_levels(order=[1, 0, 2], axis="columns")
        dataset = dataset.sort_index(axis="columns", level=0)
        dataset["tumor", "1", "t_stage"] = drawn_t_stages
        dataset["tumor", "1", "extension"] = drawn_midexts
        dataset["patient", "#", "diagnose_time"] = drawn_diag_times

        return dataset
