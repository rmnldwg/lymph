"""Module for HPV/noHPV lymphatic tumor progression models."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd

from lymph import diagnosis_times, matrix, modalities, models, types, utils

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)


class HpvWrapper(
    diagnosis_times.Composite,
    modalities.Composite,
    types.Model,
):
    """Class that models metastatic progression in HPV and non HPV lymphatic systems.

    This is achieved by creating two instances of the
    :py:class:`~lymph.models.Unilateral` model, one for the HPV+ and one for the
    HPV-.

    See Also
    --------
        :py:class:`~lymph.models.Unilateral`
            Two instances of this class are created as attributes. One for the HPV- and
            one for the noHPV model.

    """

    def __init__(
        self,
        graph_dict: types.GraphDictType,
        universal_kwargs: dict[str, Any] | None = None,
        hpv_kwargs: dict[str, Any] | None = None,
        nohpv_kwargs: dict[str, Any] | None = None,
        **_kwargs,
    ) -> None:
        """Initialize a ``unilateral`` HPV model.

        The ``graph_dict`` is a dictionary of tuples as keys and lists of strings as
        values. It is passed to both :py:class:`.models.Unilateral` instances,
        which in turn pass it to the :py:class:`.graph.Representation` class that
        stores the graph.

        """
        self._init_models(
            graph_dict=graph_dict,
            universal_kwargs=universal_kwargs,
            hpv_kwargs=hpv_kwargs,
            nohpv_kwargs=nohpv_kwargs,
        )

        diagnosis_times.Composite.__init__(
            self,
            distribution_children={"HPV": self.HPV, "noHPV": self.noHPV},
            is_distribution_leaf=False,
        )
        modalities.Composite.__init__(
            self,
            modality_children={"HPV": self.HPV, "noHPV": self.noHPV},
            is_modality_leaf=False,
        )

    def _init_models(
        self,
        graph_dict: dict[tuple[str], list[str]],
        universal_kwargs: dict[str, Any] | None = None,
        hpv_kwargs: dict[str, Any] | None = None,
        nohpv_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the two unilateral models."""
        if universal_kwargs is None:
            universal_kwargs = {}

        _hpv_kwargs = universal_kwargs.copy()
        _hpv_kwargs["graph_dict"] = graph_dict
        _hpv_kwargs.update(hpv_kwargs or {})

        _nohpv_kwargs = universal_kwargs.copy()
        _nohpv_kwargs["graph_dict"] = graph_dict
        _nohpv_kwargs.update(nohpv_kwargs or {})

        self.HPV = models.Unilateral(**_hpv_kwargs)
        self.noHPV = models.Unilateral(**_nohpv_kwargs)

        # set b_2 key name
        self.base_2_key = list(self.HPV.graph.tumors.keys())[0] + "toII"

    @classmethod
    def binary(cls, *args, **kwargs) -> HpvWrapper:
        """Initialize a binary bilateral model.

        This is a convenience method that sets the ``allowed_states`` of the
        ``uni_kwargs`` to ``[0, 1]``. All other ``args`` and ``kwargs`` are
        passed to the :py:meth:`.__init__` method.
        """
        uni_kwargs = kwargs.pop("uni_kwargs", {})
        uni_kwargs["allowed_states"] = [0, 1]
        return cls(*args, uni_kwargs=uni_kwargs, **kwargs)

    @classmethod
    def trinary(cls, *args, **kwargs) -> HpvWrapper:
        """Initialize a trinary bilateral model.

        This is a convenience method that sets the ``allowed_states`` of the
        ``uni_kwargs`` to ``[0, 1, 2]``. All other ``args`` and ``kwargs`` are
        passed to the :py:meth:`.__init__` method.
        """
        uni_kwargs = kwargs.pop("uni_kwargs", {})
        uni_kwargs["allowed_states"] = [0, 1, 2]
        return cls(*args, uni_kwargs=uni_kwargs, **kwargs)

    @property
    def is_trinary(self) -> bool:
        """Return whether the model is trinary."""
        if self.HPV.is_trinary != self.noHPV.is_trinary:
            raise ValueError("Both models must be of the same 'naryity'.")

        return self.HPV.is_trinary

    @property
    def is_binary(self) -> bool:
        """Return whether the model is binary."""
        if self.HPV.is_binary != self.noHPV.is_binary:
            raise ValueError("Both sides must be of the same 'naryity'.")

        return self.noHPV.is_binary

    def get_tumor_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model's spread from tumor to LNLs."""
        params = {
            "HPV": self.HPV.get_tumor_spread_params(as_flat=as_flat),
            "noHPV": utils.flatten(
                {
                    self.base_2_key: self.noHPV.get_tumor_spread_params(as_flat=False)[
                        self.base_2_key
                    ],
                },
            )
            if as_flat == True
            else {
                self.base_2_key: self.noHPV.get_tumor_spread_params(as_flat=False)[
                    self.base_2_key
                ],
            },
        }
        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def get_lnl_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model's spread from LNLs to tumor.

        Similarily to the :py:meth:`.get_tumor_spread_params` method.
        However, since the spread from LNLs is symmetric in HPV and noHPV,
        the spread parameters are the same and only one set is returend.
        """
        params = self.HPV.get_lnl_spread_params(as_flat=as_flat)

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def get_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model's spread edges.

        This is consistent with how the :py:meth:`.set_params`
        """
        params = self.get_tumor_spread_params(as_flat=False)

        params.update(self.get_lnl_spread_params(as_flat=as_flat))

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def get_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model.

        It returns the combination of the call to the :py:meth:`.Unilateral.get_params`
        of the HPV- and noHPV model. For the use of the ``as_dict`` and
        ``as_flat`` arguments, see the documentation of the
        :py:meth:`.types.Model.get_params` method.

        Also see the :py:meth:`.get_spread_params` method to understand how the
        symmetry settings affect the return value.
        """
        params = self.get_spread_params(as_flat=as_flat)
        params.update(self.get_distribution_params(as_flat=as_flat))

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def set_tumor_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model's spread from tumor to LNLs."""
        kwargs, global_kwargs = utils.unflatten_and_split(
            kwargs,
            expected_keys=["HPV", "noHPV"],
        )

        hpv_kwargs = global_kwargs.copy()
        hpv_kwargs.update(kwargs.get("HPV", {}))
        nohpv_kwargs = global_kwargs.copy()
        nohpv_kwargs.update(kwargs.get("noHPV", {}))

        args = self.HPV.set_tumor_spread_params(*args, **hpv_kwargs)
        args = self.noHPV.set_tumor_spread_params(*args, **nohpv_kwargs)
        utils.synchronize_params(  # might be redundant check later
            get_from=self.HPV.graph.lnl_edges,
            set_to=self.noHPV.graph.lnl_edges,
        )

        return args

    def set_lnl_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model's spread from LNLs to tumor."""
        kwargs, global_kwargs = utils.unflatten_and_split(
            kwargs,
            expected_keys=["HPV", "noHPV"],
        )

        hpv_kwargs = global_kwargs.copy()
        hpv_kwargs.update(kwargs.get("HPV", {}))
        nohpv_kwargs = global_kwargs.copy()
        nohpv_kwargs.update(kwargs.get("noHPV", {}))

        args = self.HPV.set_lnl_spread_params(*args, **hpv_kwargs)
        args = self.noHPV.set_lnl_spread_params(*args, **nohpv_kwargs)

        return args

    def set_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model's spread edges."""
        args = self.set_tumor_spread_params(*args, **kwargs)
        return self.set_lnl_spread_params(*args, **kwargs)

    def set_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set new parameters to the model."""
        args = self.set_spread_params(*args, **kwargs)
        return self.set_distribution_params(*args, **kwargs)

    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        side="ipsi",
        mapping: callable | dict[int, Any] = utils.early_late_mapping,
    ) -> None:
        """Load patient data into the model.

        Amounts to calling the :py:meth:`~lymph.models.Unilateral.load_patient_data`
        method of both sides of the neck.
        """
        self.HPV.load_patient_data(patient_data, side, mapping, True)
        self.noHPV.load_patient_data(patient_data, side, mapping, False)

    def _hmm_likelihood(self, log: bool = True, t_stage: str | None = None) -> float:
        """Compute the HMM likelihood of data, using the stored params."""
        llh = 0.0 if log else 1.0

        HPV_likelihood = self.HPV._hmm_likelihood(log=log, t_stage=t_stage)
        noHPV_likelihood = self.noHPV._hmm_likelihood(log=log, t_stage=t_stage)

        if log:
            llh += HPV_likelihood + noHPV_likelihood
        else:
            llh *= HPV_likelihood * noHPV_likelihood
        return llh

    def _bn_likelihood(self, log: bool = True, t_stage: str | None = None) -> float:
        """Compute the BN likelihood of data, using the stored params."""
        llh = 0.0 if log else 1.0

        HPV_likelihood = self.HPV._bn_likelihood(log=log, t_stage=t_stage)
        noHPV_likelihood = self.noHPV._bn_likelihood(log=log, t_stage=t_stage)

        if log:
            llh += HPV_likelihood + noHPV_likelihood
        else:
            llh *= HPV_likelihood * noHPV_likelihood
        return llh

    def likelihood(
        self,
        given_params: types.ParamsType | None = None,
        log: bool = True,
        t_stage: str | None = None,
        mode: Literal["HMM", "BN"] = "HMM",
    ):
        """Compute the (log-)likelihood of the stored data given the model (and params).

        See the documentation of :py:meth:`.types.Model.likelihood` for more
        information on how to use the ``given_params`` parameter.

        Returns the log-likelihood if ``log`` is set to ``True``. The ``mode`` parameter
        determines whether the likelihood is computed for the hidden Markov model
        (``"HMM"``) or the Bayesian network (``"BN"``).

        Note:
        ----
            The computation is much faster if no parameters are given, since then the
            transition matrix does not need to be recomputed.

        See Also:
        --------
            :py:meth:`.Unilateral.likelihood`
                The corresponding unilateral function.

        """
        try:
            # all functions and methods called here should raise a ValueError if the
            # given parameters are invalid...
            utils.safe_set_params(self, given_params)
        except ValueError:
            return -np.inf if log else 0.0

        if mode == "HMM":
            return self._hmm_likelihood(log, t_stage)
        if mode == "BN":
            return self._bn_likelihood(log, t_stage)

        raise ValueError("Invalid mode. Must be either 'HMM' or 'BN'.")

    def state_dist():
        """Does nothing, but needs to be here for the time being"""
        pass

    # everything from here is not used

    def posterior_state_dist(
        self,
        given_params: types.ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnosis: dict[str, types.DiagnosisType] | None = None,
        t_stage: str | int = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Compute joint post. dist. over ipsi & contra states, ``given_diagnosis``.

        ``given_diagnosis`` is a dictionary storing one :py:obj:`.types.DiagnosisType`
        each for the ``"ipsi"`` and ``"contra"`` side of the neck.

        Essentially, this is the risk for any possible combination of ipsi- and
        contralateral involvement, given the provided diagnosis.

        Warning:
        -------
            As in the :py:meth:`.Unilateral.posterior_state_dist` method, one may
            provide a precomputed (joint) state dist via the ``given_state_dist``
            argument (should be a square matrix). In this case, the ``given_params``
            are ignored and the model does not need to recompute e.g. the
            :py:meth:`.transition_matrix` or :py:meth:`.state_dist`, making the
            computation much faster.

            However, this will mean that ``t_stage`` and ``mode`` are also ignored,
            since these are only used to compute the state distribution.

        """
        if given_state_dist is None:
            utils.safe_set_params(self, given_params)
            given_state_dist = self.state_dist(t_stage=t_stage, mode=mode)

        if given_diagnosis is None:
            given_diagnosis = {}

        diagnosis_given_state = {}
        for side in ["ipsi", "contra"]:
            if side not in given_diagnosis:
                warnings.warn(f"No diagnosis given for {side}lateral side.")

            diagnosis_encoding = getattr(self, side).compute_encoding(
                given_diagnosis.get(side, {}),
            )
            observation_matrix = getattr(self, side).observation_matrix()
            # vector with P(Z=z|X) for each state X. A data matrix for one "patient"
            diagnosis_given_state[side] = diagnosis_encoding @ observation_matrix.T

        # matrix with P(Zi=zi,Zc=zc|Xi,Xc) * P(Xi,Xc) for all states Xi,Xc.
        joint_diagnosis_and_state = (
            np.outer(
                diagnosis_given_state["ipsi"],
                diagnosis_given_state["contra"],
            )
            * given_state_dist
        )
        # Following Bayes' theorem, this is P(Xi,Xc|Zi=zi,Zc=zc) which is given by
        # P(Zi=zi,Zc=zc|Xi,Xc) * P(Xi,Xc) / P(Zi=zi,Zc=zc)
        return joint_diagnosis_and_state / np.sum(joint_diagnosis_and_state)

    def marginalize(
        self,
        involvement: dict[str, types.PatternType],
        given_state_dist: np.ndarray | None = None,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> float:
        """Marginalize ``given_state_dist`` over matching ``involvement`` patterns.

        Any state that matches the provided ``involvement`` pattern is marginalized
        over. For this, the :py:func:`.matrix.compute_encoding` function is used.

        If ``given_state_dist`` is ``None``, it will be computed by calling
        :py:meth:`.state_dist` with the given ``t_stage`` and ``mode``. These arguments
        are ignored if ``given_state_dist`` is provided.
        """
        if given_state_dist is None:
            given_state_dist = self.state_dist(t_stage=t_stage, mode=mode)

        marginalize_over_states = {}
        for side in ["ipsi", "contra"]:
            side_graph = getattr(self, side).graph
            marginalize_over_states[side] = matrix.compute_encoding(
                lnls=side_graph.lnls.keys(),
                pattern=involvement.get(side, {}),
                base=3 if self.is_trinary else 2,
            )
        return (
            marginalize_over_states["ipsi"]
            @ given_state_dist
            @ marginalize_over_states["contra"]
        )

    def risk(
        self,
        involvement: dict[str, types.PatternType],
        given_params: types.ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnosis: dict[str, types.DiagnosisType] | None = None,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> float:
        """Compute risk of the ``involvement`` patterns, given parameters and diagnosis.

        The ``involvement`` of interest is expected to be a :py:obj:`.PatternType` for
        each side of the neck (``"ipsi"`` and ``"contra"``). This method then
        marginalizes over those posterior state probabilities that match the
        ``involvement`` patterns.

        If ``involvement`` is not provided, the method returns the posterior state
        distribution as computed by the :py:meth:`.posterior_state_dist` method. See
        its docstring for more details on the remaining arguments.
        """
        # TODO: test this method
        posterior_state_dist = self.posterior_state_dist(
            given_params=given_params,
            given_state_dist=given_state_dist,
            given_diagnosis=given_diagnosis,
            t_stage=t_stage,
            mode=mode,
        )

        return self.marginalize(involvement, posterior_state_dist)

    def draw_patients(
        self,
        num: int,
        stage_dist: Iterable[float],
        rng: np.random.Generator | None = None,
        seed: int = 42,
        **_kwargs,
    ) -> pd.DataFrame:
        """Draw ``num`` random patients from the parametrized model.

        See Also
        --------
            :py:meth:`.diagnosis_times.Distribution.draw_diag_times`
                Method to draw diagnosis times from a distribution.
            :py:meth:`.Unilateral.draw_diagnosis`
                Method to draw individual diagnosis from a unilateral model.
            :py:meth:`.Unilateral.draw_patients`
                The unilateral method to draw a synthetic dataset.

        """
        if rng is None:
            rng = np.random.default_rng(seed)

        if sum(stage_dist) != 1.0:
            warnings.warn("Sum of stage distribution is not 1. Renormalizing.")
            stage_dist = np.array(stage_dist) / sum(stage_dist)

        drawn_t_stages = rng.choice(
            a=self.t_stages,
            p=stage_dist,
            size=num,
        )
        drawn_diag_times = [
            self.get_distribution(t_stage).draw_diag_times(rng=rng)
            for t_stage in drawn_t_stages
        ]

        drawn_obs_ipsi = self.ipsi.draw_diagnosis(drawn_diag_times, rng=rng)
        drawn_obs_noHPV = self.contra.draw_diagnosis(drawn_diag_times, rng=rng)
        drawn_obs = np.concatenate([drawn_obs_ipsi, drawn_obs_noHPV], axis=1)

        # construct MultiIndex with "ipsi" and "contra" at top level to allow
        # concatenation of the two separate drawn diagnosis
        sides = ["ipsi", "contra"]
        modality_names = list(self.get_all_modalities().keys())
        lnl_names = list(self.ipsi.graph.lnls.keys())
        multi_cols = pd.MultiIndex.from_product([sides, modality_names, lnl_names])

        # reorder the column levels and thus also the individual columns to match the
        # LyProX format without mixing up the data
        dataset = pd.DataFrame(drawn_obs, columns=multi_cols)
        dataset = dataset.reorder_levels(order=[1, 0, 2], axis="columns")
        dataset = dataset.sort_index(axis="columns", level=0)
        dataset[("tumor", "1", "t_stage")] = drawn_t_stages

        return dataset
