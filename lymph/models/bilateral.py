from __future__ import annotations

import logging
import warnings
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from lymph import diagnose_times, matrix, modalities, models, types, utils

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)


class Bilateral(
    diagnose_times.Composite,
    modalities.Composite,
    types.Model,
):
    """Class that models metastatic progression in a bilateral lymphatic system.

    This is achieved by creating two instances of the
    :py:class:`~lymph.models.Unilateral` model, one for the ipsi- and one for the
    contralateral side of the neck. The two sides are assumed to be independent of each
    other, given the diagnose time over which we marginalize.

    See Also:
        :py:class:`~lymph.models.Unilateral`
            Two instances of this class are created as attributes. One for the ipsi- and
            one for the contralateral side of the neck.
    """
    def __init__(
        self,
        graph_dict: types.GraphDictType,
        is_symmetric: dict[str, bool] | None = None,
        uni_kwargs: dict[str, Any] | None = None,
        ipsi_kwargs: dict[str, Any] | None = None,
        contra_kwargs: dict[str, Any] | None = None,
        **_kwargs,
    ) -> None:
        """Initialize both sides of the neck as :py:class:`.models.Unilateral`.

        The ``graph_dict`` is a dictionary of tuples as keys and lists of strings as
        values. It is passed to both :py:class:`.models.Unilateral` instances,
        which in turn pass it to the :py:class:`.graph.Representation` class that
        stores the graph.

        With the dictionary ``is_symmetric`` the user can specify which aspects of the
        model are symmetric. Valid keys are ``"tumor_spread"`` and ``"lnl_spread"``.
        The values are booleans, with ``True`` meaning that the aspect is symmetric.

        Note:
            The symmetries of tumor and LNL spread are only guaranteed if the
            respective parameters are set via the :py:meth:`.set_params()` method of
            this bilateral model. It is still possible to set different parameters for
            the ipsi- and contralateral side by using their respective
            :py:meth:`.Unilateral.set_params()` method.

        The ``uni_kwargs`` are passed to both instances of the unilateral model, while
        the ``ipsi_kwargs`` and ``contra_kwargs`` are passed to the ipsi- and
        contralateral side, respectively. The ipsi- and contralateral kwargs override
        the unilateral kwargs and may also override the ``graph_dict``. This allows the
        user to specify different graphs for the two sides of the neck.
        """
        self._init_models(
            graph_dict=graph_dict,
            uni_kwargs=uni_kwargs,
            ipsi_kwargs=ipsi_kwargs,
            contra_kwargs=contra_kwargs,
        )

        if is_symmetric is None:
            is_symmetric = {}

        is_symmetric["tumor_spread"] = is_symmetric.get("tumor_spread", False)
        is_symmetric["lnl_spread"] = is_symmetric.get("lnl_spread", True)

        self.is_symmetric = is_symmetric

        diagnose_times.Composite.__init__(
            self,
            distribution_children={"ipsi": self.ipsi, "contra": self.contra},
            is_distribution_leaf=False,
        )
        modalities.Composite.__init__(
            self,
            modality_children={"ipsi": self.ipsi, "contra": self.contra},
            is_modality_leaf=False,
        )


    def _init_models(
        self,
        graph_dict: dict[tuple[str], list[str]],
        uni_kwargs: dict[str, Any] | None = None,
        ipsi_kwargs: dict[str, Any] | None = None,
        contra_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the two unilateral models."""
        if uni_kwargs is None:
            uni_kwargs = {}

        _ipsi_kwargs = uni_kwargs.copy()
        _ipsi_kwargs["graph_dict"] = graph_dict
        _ipsi_kwargs.update(ipsi_kwargs or {})

        _contra_kwargs = uni_kwargs.copy()
        _contra_kwargs["graph_dict"] = graph_dict
        _contra_kwargs.update(contra_kwargs or {})

        self.ipsi   = models.Unilateral(**_ipsi_kwargs)
        self.contra = models.Unilateral(**_contra_kwargs)


    @classmethod
    def binary(cls, *args, **kwargs) -> Bilateral:
        """Initialize a binary bilateral model.

        This is a convenience method that sets the ``allowed_states`` of the
        ``uni_kwargs`` to ``[0, 1]``. All other ``args`` and ``kwargs`` are
        passed to the :py:meth:`.__init__` method.
        """
        uni_kwargs = kwargs.pop("uni_kwargs", {})
        uni_kwargs["allowed_states"] = [0, 1]
        return cls(*args, uni_kwargs=uni_kwargs, **kwargs)

    @classmethod
    def trinary(cls, *args, **kwargs) -> Bilateral:
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
        if self.ipsi.is_trinary != self.contra.is_trinary:
            raise ValueError("Both sides must be of the same 'naryity'.")

        return self.ipsi.is_trinary

    @property
    def is_binary(self) -> bool:
        """Return whether the model is binary."""
        if self.ipsi.is_binary != self.contra.is_binary:
            raise ValueError("Both sides must be of the same 'naryity'.")

        return self.ipsi.is_binary


    def get_tumor_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model's spread from tumor to LNLs.

        If the attribute dictionary :py:attr:`.is_symmetric` stores the key-value pair
        ``"tumor_spread": True``, the parameters are returned as a single dictionary,
        since they are the same ipsi- and contralaterally. Otherwise, the parameters
        are returned as a dictionary with two keys, ``"ipsi"`` and ``"contra"``.
        """
        params = {
            "ipsi": self.ipsi.get_tumor_spread_params(as_flat=as_flat),
            "contra": self.contra.get_tumor_spread_params(as_flat=as_flat),
        }

        if self.is_symmetric["tumor_spread"]:
            if params["ipsi"] != params["contra"]:
                warnings.warn(
                    "The tumor spread parameters are not symmetric. "
                    "Returning the ipsilateral parameters."
                )

            params = params["ipsi"]

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()


    def get_lnl_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model's spread from LNLs to tumor.

        Similarily to the :py:meth:`.get_tumor_spread_params` method, this returns only
        one dictionary if the attribute dictionary :py:attr:`.is_symmetric` stores the
        key-value pair ``"lnl_spread": True``. Otherwise, the parameters are returned
        as a dictionary with two keys, ``"ipsi"`` and ``"contra"``.
        """
        params = {
            "ipsi": self.ipsi.get_lnl_spread_params(as_flat=as_flat),
            "contra": self.contra.get_lnl_spread_params(as_flat=as_flat),
        }

        if self.is_symmetric["lnl_spread"]:
            if params["ipsi"] != params["contra"]:
                warnings.warn(
                    "The LNL spread parameters are not symmetric. "
                    "Returning the ipsilateral parameters."
                )

            params = params["ipsi"]

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()


    def get_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model's spread edges.

        Depending on the symmetries (i.e. the ``is_symmetric`` attribute), this returns
        different results:

        If ``is_symmetric["tumor_spread"] = False``, the flattened (``as_flat=True``)
        dictionary (``as_dict=True``) will contain keys of the form
        ``ipsi_Tto<lnl>_spread`` and ``contra_Tto<lnl>_spread``, where ``<lnl>`` is the
        name of the lymph node level. However, if the tumor spread is set to be
        symmetric, the leading ``ipsi_`` or ``contra_`` is omitted, since it's valid
        for both sides.

        This is consistent with how the :py:meth:`.set_params`
        method expects the keyword arguments in case of the symmetry configurations.

        >>> model = Bilateral(graph_dict={
        ...     ("tumor", "T"): ["II", "III"],
        ...     ("lnl", "II"): ["III"],
        ...     ("lnl", "III"): [],
        ... })
        >>> num_dims = model.get_num_dims()
        >>> model.set_spread_params(*np.round(np.linspace(0., 1., num_dims+1), 2))
        (1.0,)
        >>> model.get_spread_params(as_flat=False)   # doctest: +NORMALIZE_WHITESPACE
        {'ipsi':    {'TtoII': {'spread': 0.0},
                     'TtoIII': {'spread': 0.2}},
         'contra':  {'TtoII': {'spread': 0.4},
                     'TtoIII': {'spread': 0.6}},
         'IItoIII': {'spread': 0.8}}
        >>> model.get_spread_params(as_flat=True)    # doctest: +NORMALIZE_WHITESPACE
        {'ipsi_TtoII_spread': 0.0,
         'ipsi_TtoIII_spread': 0.2,
         'contra_TtoII_spread': 0.4,
         'contra_TtoIII_spread': 0.6,
         'IItoIII_spread': 0.8}
        """
        params = self.get_tumor_spread_params(as_flat=False)

        if not self.is_symmetric["tumor_spread"] and not self.is_symmetric["lnl_spread"]:
            params["ipsi"].update(self.get_lnl_spread_params(as_flat=False)["ipsi"])
            params["contra"].update(self.get_lnl_spread_params(as_flat=False)["contra"])
        else:
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
        of the ipsi- and contralateral side. For the use of the ``as_dict`` and
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
        kwargs, global_kwargs = utils.unflatten_and_split(kwargs, expected_keys=["ipsi", "contra"])

        ipsi_kwargs = global_kwargs.copy()
        ipsi_kwargs.update(kwargs.get("ipsi", {}))
        contra_kwargs = global_kwargs.copy()
        contra_kwargs.update(kwargs.get("contra", {}))

        args = self.ipsi.set_tumor_spread_params(*args, **ipsi_kwargs)
        if self.is_symmetric["tumor_spread"]:
            utils.synchronize_params(
                get_from=self.ipsi.graph.tumor_edges,
                set_to=self.contra.graph.tumor_edges,
            )
        else:
            args = self.contra.set_tumor_spread_params(*args, **contra_kwargs)

        return args


    def set_lnl_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model's spread from LNLs to tumor."""
        kwargs, global_kwargs = utils.unflatten_and_split(
            kwargs, expected_keys=["ipsi", "contra"],
        )

        ipsi_kwargs = global_kwargs.copy()
        ipsi_kwargs.update(kwargs.get("ipsi", {}))
        contra_kwargs = global_kwargs.copy()
        contra_kwargs.update(kwargs.get("contra", {}))

        args = self.ipsi.set_lnl_spread_params(*args, **ipsi_kwargs)
        if self.is_symmetric["lnl_spread"]:
            utils.synchronize_params(
                get_from=self.ipsi.graph.lnl_edges,
                set_to=self.contra.graph.lnl_edges,
            )
        else:
            args = self.contra.set_lnl_spread_params(*args, **contra_kwargs)

        return args


    def set_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model's spread edges."""
        args = self.set_tumor_spread_params(*args, **kwargs)
        return self.set_lnl_spread_params(*args, **kwargs)


    def set_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set new parameters to the model.

        This works almost exactly as the unilateral model's
        :py:meth:`.Unilateral.set_params` method. However, this one allows the user to
        set the parameters of individual sides of the neck by prefixing the keyword
        arguments' names with ``"ipsi_"`` or ``"contra_"``.

        Anything not prefixed by ``"ipsi_"`` or ``"contra_"`` is passed to both sides
        of the neck. This does obviously not work with positional arguments.

        When setting the parameters via positional arguments, the order is
        important:

        1. The parameters of the edges from tumor to LNLs:

           1. first the ipsilateral parameters,
           2. if ``is_symmetric["tumor_spread"]`` is ``False``, the contralateral
              parameters. Otherwise, the ipsilateral parameters are used for both
              sides.

        2. The parameters of the edges from LNLs to tumor:

           1. again, first the ipsilateral parameters,
           2. if ``is_symmetric["lnl_spread"]`` is ``False``, the contralateral
              parameters. Otherwise, the ipsilateral parameters are used for both
              sides.

        3. The parameters of the parametric distributions for marginalizing over
           diagnose times.

        When still some positional arguments remain after that, they are returned
        in a tuple.
        """
        args = self.set_spread_params(*args, **kwargs)
        return self.set_distribution_params(*args, **kwargs)


    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        mapping: callable | dict[int, Any] = utils.early_late_mapping,
    ) -> None:
        """Load patient data into the model.

        This amounts to calling the :py:meth:`~lymph.models.Unilateral.load_patient_data`
        method of both sides of the neck.
        """
        self.ipsi.load_patient_data(patient_data, "ipsi", mapping)
        self.contra.load_patient_data(patient_data, "contra", mapping)


    def state_dist(
        self,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Compute the joint distribution over the ipsi- & contralateral hidden states.

        This computes the state distributions of both sides and returns their outer
        product. In case ``mode`` is ``"HMM"`` (default), the state distributions are
        first marginalized over the diagnose time distribtions of the respective
        ``t_stage``.

        See Also:
            :py:meth:`.Unilateral.state_dist`
                The corresponding unilateral function. Note that this method returns
                a 2D array, because it computes the probability of any possible
                combination of ipsi- and contralateral states.
        """
        if mode == "HMM":
            ipsi_state_evo = self.ipsi.state_dist_evo()
            contra_state_evo = self.contra.state_dist_evo()
            time_marg_matrix = np.diag(self.get_distribution(t_stage).pmf)
            result = ipsi_state_evo.T @ time_marg_matrix @ contra_state_evo

        elif mode == "BN":
            ipsi_state_dist = self.ipsi.state_dist(mode=mode)
            contra_state_dist = self.contra.state_dist(mode=mode)
            result = np.outer(ipsi_state_dist, contra_state_dist)

        else:
            raise ValueError(f"Unknown mode '{mode}'.")

        return result


    def obs_dist(
        self,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Compute the joint distribution over the ipsi- & contralateral observations.

        See Also:
            :py:meth:`.Unilateral.obs_dist`
                The corresponding unilateral function. Note that this method returns
                a 2D array, because it computes the probability of any possible
                combination of ipsi- and contralateral observations.
        """
        joint_state_dist = self.state_dist(t_stage=t_stage, mode=mode)
        return (
            self.ipsi.observation_matrix().T
            @ joint_state_dist
            @ self.contra.observation_matrix()
        )


    def patient_likelihoods(
        self,
        t_stage: str,
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Compute the likelihood of each patient individually."""
        joint_state_dist = self.state_dist(t_stage=t_stage, mode=mode)
        return matrix.fast_trace(
            self.ipsi.diagnose_matrix(t_stage),
            joint_state_dist @ self.contra.diagnose_matrix(t_stage).T,
        )


    def _bn_likelihood(self, log: bool = True, t_stage: str | None = None) -> float:
        """Compute the BN likelihood of data, using the stored params."""
        joint_state_dist = self.state_dist(mode="BN")
        patient_llhs = matrix.fast_trace(
            self.ipsi.diagnose_matrix(t_stage),
            joint_state_dist @ self.contra.diagnose_matrix(t_stage).T,
        )

        return np.sum(np.log(patient_llhs)) if log else np.prod(patient_llhs)


    def _hmm_likelihood(self, log: bool = True, t_stage: str | None = None) -> float:
        """Compute the HMM likelihood of data, using the stored params."""
        llh = 0. if log else 1.

        ipsi_dist_evo = self.ipsi.state_dist_evo()
        contra_dist_evo = self.contra.state_dist_evo()

        if t_stage is None:
            t_stages = self.t_stages
        else:
            t_stages = [t_stage]

        for stage in t_stages:
            diag_time_matrix = np.diag(self.get_distribution(stage).pmf)

            # Note that I am not using the `comp_joint_state_dist` method here, since
            # that would recompute the state dist evolution for each T-stage.
            joint_state_dist = (
                ipsi_dist_evo.T
                @ diag_time_matrix
                @ contra_dist_evo
            )
            patient_llhs = matrix.fast_trace(
                self.ipsi.diagnose_matrix(stage),
                joint_state_dist @ self.contra.diagnose_matrix(stage).T,
            )
            llh = utils.add_or_mult(llh, patient_llhs, log)

        return llh


    def likelihood(
        self,
        given_params: types.ParamsType | None = None,
        log: bool = True,
        mode: Literal["HMM", "BN"] = "HMM",
        for_t_stage: str | None = None,
    ):
        """Compute the (log-)likelihood of the stored data given the model (and params).

        See the documentation of :py:meth:`.types.Model.likelihood` for more
        information on how to use the ``given_params`` parameter.

        Returns the log-likelihood if ``log`` is set to ``True``. The ``mode`` parameter
        determines whether the likelihood is computed for the hidden Markov model
        (``"HMM"``) or the Bayesian network (``"BN"``).

        Note:
            The computation is much faster if no parameters are given, since then the
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
        if mode == "BN":
            return self._bn_likelihood(log, for_t_stage)

        raise ValueError("Invalid mode. Must be either 'HMM' or 'BN'.")


    def posterior_state_dist(
        self,
        given_params: types.ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnoses: dict[str, types.DiagnoseType] | None = None,
        t_stage: str | int = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Compute joint post. dist. over ipsi & contra states, ``given_diagnoses``.

        The ``given_diagnoses`` is a dictionary storing one :py:obj:`.types.DiagnoseType`
        each for the ``"ipsi"`` and ``"contra"`` side of the neck.

        Essentially, this is the risk for any possible combination of ipsi- and
        contralateral involvement, given the provided diagnoses.

        Warning:
            As in the :py:meth:`.Unilateral.posterior_state_dist` method, one may
            provide a precomputed (joint) state distribution via the ``given_state_dist``
            argument (should be a square matric). In this case, the ``given_params``
            are ignored and the model does not need to recompute e.g. the
            :py:meth:`.transition_matrix` or :py:meth:`.state_dist`, making the
            computation much faster.

            However, this will mean that ``t_stage`` and ``mode`` are also ignored,
            since these are only used to compute the state distribution.
        """
        if given_state_dist is None:
            utils.safe_set_params(self, given_params)
            given_state_dist = self.state_dist(t_stage=t_stage, mode=mode)

        if given_diagnoses is None:
            given_diagnoses = {}

        diagnose_given_state = {}
        for side in ["ipsi", "contra"]:
            if side not in given_diagnoses:
                warnings.warn(f"No diagnoses given for {side}lateral side.")

            diagnose_encoding = getattr(self, side).compute_encoding(
                given_diagnoses.get(side, {})
            )
            observation_matrix = getattr(self, side).observation_matrix()
            # vector with P(Z=z|X) for each state X. A data matrix for one "patient"
            diagnose_given_state[side] = diagnose_encoding @ observation_matrix.T

        # matrix with P(Zi=zi,Zc=zc|Xi,Xc) * P(Xi,Xc) for all states Xi,Xc.
        joint_diagnose_and_state = np.outer(
            diagnose_given_state["ipsi"],
            diagnose_given_state["contra"],
        ) * given_state_dist
        # Following Bayes' theorem, this is P(Xi,Xc|Zi=zi,Zc=zc) which is given by
        # P(Zi=zi,Zc=zc|Xi,Xc) * P(Xi,Xc) / P(Zi=zi,Zc=zc)
        return joint_diagnose_and_state / np.sum(joint_diagnose_and_state)


    def risk(
        self,
        involvement: dict[str, types.PatternType] | None = None,
        given_params: types.ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnoses: dict[str, types.DiagnoseType] | None = None,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> float:
        """Compute risk of the ``involvement`` patterns, given parameters and diagnoses.

        The ``involvement`` of interest is expected to be a :py:obj:`.PatternType` for
        each side of the neck (``"ipsi"`` and ``"contra"``). This method then
        marginalizes over those posterior state probabilities that match the
        ``involvement`` patterns.

        If ``involvement`` is not provided, the method returns the posterior state
        distribution as computed by the :py:meth:`.posterior_state_dist` method. See
        its docstring for more details on the remaining arguments.
        """
        # TODO: test this method
        posterior_state_probs = self.posterior_state_dist(
            given_params=given_params,
            given_state_dist=given_state_dist,
            given_diagnoses=given_diagnoses,
            t_stage=t_stage,
            mode=mode,
        )

        if involvement is None:
            return posterior_state_probs

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
            @ posterior_state_probs
            @ marginalize_over_states["contra"]
        )


    def draw_patients(
        self,
        num: int,
        stage_dist: Iterable[float],
        rng: np.random.Generator | None = None,
        seed: int = 42,
        **_kwargs,
    ) -> pd.DataFrame:
        """Draw ``num`` random patients from the parametrized model.

        See Also:
            :py:meth:`.diagnose_times.Distribution.draw_diag_times`
                Method to draw diagnose times from a distribution.
            :py:meth:`.Unilateral.draw_diagnoses`
                Method to draw individual diagnoses from a unilateral model.
            :py:meth:`.Unilateral.draw_patients`
                The unilateral method to draw a synthetic dataset.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        if sum(stage_dist) != 1.:
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

        drawn_obs_ipsi = self.ipsi.draw_diagnoses(drawn_diag_times, rng=rng)
        drawn_obs_contra = self.contra.draw_diagnoses(drawn_diag_times, rng=rng)
        drawn_obs = np.concatenate([drawn_obs_ipsi, drawn_obs_contra], axis=1)

        # construct MultiIndex with "ipsi" and "contra" at top level to allow
        # concatenation of the two separate drawn diagnoses
        sides = ["ipsi", "contra"]
        modality_names = list(self.get_all_modalities().keys())
        lnl_names = [lnl for lnl in self.ipsi.graph.lnls.keys()]
        multi_cols = pd.MultiIndex.from_product([sides, modality_names, lnl_names])

        # reorder the column levels and thus also the individual columns to match the
        # LyProX format without mixing up the data
        dataset = pd.DataFrame(drawn_obs, columns=multi_cols)
        dataset = dataset.reorder_levels(order=[1, 0, 2], axis="columns")
        dataset = dataset.sort_index(axis="columns", level=0)
        dataset[('tumor', '1', 't_stage')] = drawn_t_stages

        return dataset
