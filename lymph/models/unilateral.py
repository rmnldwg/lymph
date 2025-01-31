"""Base module for the lymphatic system models."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd
from cachetools import LRUCache

from lymph import diagnosis_times, graph, matrix, modalities, types, utils
from lymph.utils import (
    add_or_mult,
    dict_to_func,  # noqa: F401
    draw_diagnosis,  # noqa: F401
    early_late_mapping,
    flatten,
    get_params_from,
    set_params_for,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


MAP_T_COL = ("_model", "#", "t_stage")
RAW_T_COL = ("tumor", "1", "t_stage")


class Unilateral(
    diagnosis_times.Composite,
    modalities.Composite,
    types.Model,
):
    """Class that models metastatic progression in a unilateral lymphatic system.

    It does this by representing it as a directed graph (DAG), which is stored in and
    managed by the attribute :py:attr:`~graph`. The progression itself can be modelled
    via hidden Markov models (HMM) or Bayesian networks (BN). In both cases, instances
    of this class allow to calculate the probability of a certain hidden pattern of
    involvement, given an individual diagnosis of a patient.
    """

    def __init__(
        self,
        graph_dict: types.GraphDictType,
        named_params: Sequence[str] | None = None,
        tumor_state: int | None = None,
        allowed_states: list[int] | None = None,
        max_time: int = 10,
        **_kwargs,
    ) -> None:
        """Create a new instance of the :py:class:`~Unilateral` class.

        The ``graph_dict`` that represents the lymphatic system should given as a
        dictionary. Its keys are tuples of the form ``("tumor", "<tumor_name>")`` or
        ``("lnl", "<lnl_name>")``. The values are lists of strings that represent the
        names of the nodes that are connected to the node given by the key.

        .. note::

            Do make sure the values in the dictionary are of type ``list`` and *not*
            ``set``. Sets do not preserve the order of the elements and thus the order
            of the edges in the graph. This may lead to inconsistencies in the model.

        For example, the following graph represents a lymphatic system with one tumors
        and three lymph node levels:

        .. code-block:: python

            graph = {
                ("tumor", "T"): ["II", "III", "IV"],
                ("lnl", "II"): ["III"],
                ("lnl", "III"): ["IV"],
                ("lnl", "IV"): [],
            }

        The ``tumor_state`` is the initial (and unchangeable) state of the tumor.
        Typically, this can be omitted and is then set to be the maximum of the
        ``allowed_states``, which is the states the LNLs can take on. The default is a
        binary representation with ``allowed_states=[0, 1]``. For this, one can also
        use the classmethod :py:meth:`~Unilateral.binary`. For a trinary representation
        with ``allowed_states=[0, 1, 2]`` use the classmethod
        :py:meth:`~Unilateral.trinary`.

        The ``max_time`` parameter defines the latest possible time step for a
        diagnosis. In the HMM case, the probability distribution over all hidden states
        is evolved from :math:`t=0` to ``max_time``. In the BN case, this parameter has
        no effect.

        """
        self.graph = graph.Representation(
            graph_dict=graph_dict,
            tumor_state=tumor_state,
            allowed_states=allowed_states,
        )

        if named_params is not None:
            self.named_params = named_params

        diagnosis_times.Composite.__init__(
            self,
            max_time=max_time,
            is_distribution_leaf=True,
        )
        modalities.Composite.__init__(self, is_modality_leaf=True)
        self._patient_data: pd.DataFrame | None = None
        self._cache_version: int = 0
        self._data_matrix_cache: LRUCache = LRUCache(maxsize=64)
        self._diagnosis_matrix_cache: LRUCache = LRUCache(maxsize=64)

    @classmethod
    def binary(cls, graph_dict: types.GraphDictType, **kwargs) -> Unilateral:
        """Create an instance of the :py:class:`~Unilateral` class with binary LNLs."""
        return cls(graph_dict, allowed_states=[0, 1], **kwargs)

    @classmethod
    def trinary(cls, graph_dict: types.GraphDictType, **kwargs) -> Unilateral:
        """Create an instance of the :py:class:`~Unilateral` class with trinary LNLs."""
        return cls(graph_dict, allowed_states=[0, 1, 2], **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        return (
            f"{type(self).__name__}("
            f"graph_dict={self.graph.to_dict()}, "
            f"tumor_state={list(self.graph.tumors.values())[0].state}, "
            f"allowed_states={self.graph.allowed_states}, "
            f"max_time={self.max_time})"
        )

    def __str__(self) -> str:
        """Print info about the instance."""
        return (
            f"Unilateral with {len(self.graph.tumors)} tumors "
            f"and {len(self.graph.lnls)} LNLs"
        )

    @property
    def is_trinary(self) -> bool:
        """Return whether the model is trinary."""
        return self.graph.is_trinary

    @property
    def is_binary(self) -> bool:
        """Return whether the model is binary."""
        return self.graph.is_binary

    def get_t_stages(
        self,
        which: Literal["valid", "distributions", "data"] = "valid",
    ) -> list[str]:
        """Return the T-stages of the model."""
        if which in ("valid", "distributions"):
            distribution_t_stages = super().t_stages
            if which == "distributions":
                return distribution_t_stages

        if which in ("valid", "data"):
            try:
                data_t_stages = self.patient_data[MAP_T_COL].unique()
            except AttributeError:
                data_t_stages = []

            if which == "data":
                return data_t_stages

        if which == "valid":
            return sorted(set(distribution_t_stages) & set(data_t_stages))

        raise ValueError(
            f"Invalid value for 'which': {which}. Must be either 'valid', "
            "'distributions', or 'data'.",
        )

    def get_tumor_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Get the parameters of the tumor spread edges."""
        return get_params_from(self.graph.tumor_edges, as_dict, as_flat)

    def get_lnl_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Get the parameters of the LNL spread edges.

        In the trinary case, this includes the growth parameters as well as the
        microscopic modification parameters.
        """
        return get_params_from(self.graph.lnl_edges, as_dict, as_flat)

    def get_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Get the parameters of the spread edges."""
        params = self.get_tumor_spread_params(as_flat=as_flat)
        params.update(self.get_lnl_spread_params(as_flat=as_flat))

        if as_flat or not as_dict:
            params = flatten(params)

        return params if as_dict else params.values()

    def get_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Get the parameters of the model.

        If ``as_dict`` is ``True``, the parameters are returned as a dictionary. If
        ``as_flat`` is ``True``, the dictionary is flattened, i.e., all nested
        dictionaries are merged into one, using :py:func:`~lymph.helper.flatten`.
        """
        params = self.get_spread_params(as_flat=as_flat)
        params.update(self.get_distribution_params(as_flat=as_flat))

        if as_flat or not as_dict:
            params = flatten(params)

        return params if as_dict else params.values()

    def set_tumor_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Assign new parameters to the tumor spread edges."""
        return set_params_for(self.graph.tumor_edges, *args, **kwargs)

    def set_lnl_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Assign new parameters to the LNL spread edges."""
        return set_params_for(self.graph.lnl_edges, *args, **kwargs)

    def set_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Assign new parameters to the spread edges."""
        args = self.set_tumor_spread_params(*args, **kwargs)
        return self.set_lnl_spread_params(*args, **kwargs)

    def set_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Assign new parameters to the model.

        The parameters can be provided either via positional arguments or via keyword
        arguments. The positional arguments are used up one by one first by the
        :py:meth:`lymph.graph.set_params` method and then by the
        :py:meth:`lymph.models.Unilateral.set_distribution_params` method.

        The keyword arguments can be of the format ``"<edge_name>_<param_name>"`` or
        ``"<t_stage>_<param_name>"`` for the distributions over diagnosis times. If only
        a ``"<param_name>"`` is provided, it is assumed to be a global parameter and is
        sent to all edges or distributions. But the more specific keyword arguments
        override the global ones, which in turn override the positional arguments.

        >>> graph = {
        ...     ("tumor", "T"): ["II", "III"],
        ...     ("lnl", "II"): ["III"],
        ...     ("lnl", "III"): [],
        ... }
        >>> model = Unilateral.trinary(
        ...     graph_dict=graph,
        ...     is_micro_mod_shared=True,
        ...     is_growth_shared=True,
        ... )
        >>> model.set_params(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.99, AtoB_param="not_used")
        (0.99,)
        >>> model.get_params(as_dict=True)  # doctest: +NORMALIZE_WHITESPACE
        {'TtoII_spread': 0.1,
         'TtoIII_spread': 0.2,
         'II_growth': 0.3,
         'IItoIII_spread': 0.4,
         'IItoIII_micro': 0.5,
         'III_growth': 0.6}
        >>> _ = model.set_params(growth=0.123)
        >>> model.get_params(as_dict=True)  # doctest: +NORMALIZE_WHITESPACE
        {'TtoII_spread': 0.1,
         'TtoIII_spread': 0.2,
         'II_growth': 0.123,
         'IItoIII_spread': 0.4,
         'IItoIII_micro': 0.5,
         'III_growth': 0.123}
        """
        args = self.set_spread_params(*args, **kwargs)
        return self.set_distribution_params(*args, **kwargs)

    def transition_prob(self, new_state: list[int], assign: bool = False) -> float:
        """Compute probability to transition to ``new_state``, given its current state.

        The probability is computed as the product of the transition probabilities of
        the individual LNLs. If ``assign`` is ``True``, the new state is assigned to
        the model using the method :py:meth:`lymph.graph.Representation.set_state`.
        """
        trans_prob = 1
        for i, lnl in enumerate(self.graph.lnls):
            trans_prob *= lnl.comp_trans_prob(new_state=new_state[i])
            if trans_prob == 0:
                break

        if assign:
            self.graph.set_state(new_state)

        return trans_prob

    def diagnosis_prob(
        self,
        diagnosis: pd.Series | dict[str, dict[str, bool]],
    ) -> float:
        """Compute the probability to observe a diagnosis given the current state.

        The ``diagnosis`` is either a pandas ``Series`` object corresponding to one row
        of a patient data table, or a dictionary with keys of diagnostic modalities and
        values of dictionaries holding the observation for each LNL under the
        respective key.

        It returns the probability of observing this particular combination of
        diagnosis, given the current state of the system.
        """
        prob = 1.0
        for name, modality in self.get_all_modalities().items():
            if name in diagnosis:
                mod_diagnosis = diagnosis[name]
                for lnl in self.graph.lnls:
                    try:
                        lnl_diagnosis = mod_diagnosis[lnl.name]
                    except KeyError:
                        continue
                    except IndexError as idx_err:
                        raise ValueError(
                            "diagnosis were not provided in the correct format",
                        ) from idx_err
                    prob *= lnl.comp_obs_prob(lnl_diagnosis, modality.confusion_matrix)
        return prob

    @property
    def obs_list(self):
        """Return the list of all possible observations.

        They are ordered like the :py:attr:`.graph.Representation.state_list`, but
        additionally by modality. E.g., for two LNLs II, III and two modalities CT,
        pathology, the list would look like this:

        >>> model = Unilateral(graph_dict={
        ...     ("tumor", "T"): ["II" , "III"],
        ...     ("lnl", "II"): ["III"],
        ...     ("lnl", "III"): [],
        ... })
        >>> model.set_modality("CT", spec=0.8, sens=0.8)
        >>> model.set_modality("pathology", spec=1.0, sens=1.0)
        >>> model.obs_list  # doctest: +ELLIPSIS
        array([[0, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0],
               [0, 0, 1, 1],
               ...
               [1, 1, 0, 1],
               [1, 1, 1, 0],
               [1, 1, 1, 1]])

        The first two columns correspond to the observation of LNLs II and III under
        modality CT, the second two columns correspond to the same LNLs under the
        pathology modality.
        """
        possible_obs_list = []
        for modality in self.get_all_modalities().values():
            possible_obs = np.arange(modality.confusion_matrix.shape[1])
            for _ in self.graph.lnls:
                possible_obs_list.append(possible_obs.copy())

        return np.array(list(product(*possible_obs_list)))

    def transition_matrix(self) -> np.ndarray:
        r"""Matrix encoding the probabilities to transition from one state to another.

        This is the crucial object for modelling the evolution of the probabilistic
        system in the context of the hidden Markov model. It has the shape
        :math:`2^N \\times 2^N` where :math:`N` is the number of nodes in the graph.
        The :math:`i`-th row and :math:`j`-th column encodes the probability to
        transition from the :math:`i`-th state to the :math:`j`-th state. The states
        are ordered as in the :py:attr:`.graph.Representation.state_list`.

        .. seealso::

            :py:func:`~lymph.descriptors.matrix.generate_transition`
                The function actually computing the transition matrix.

        >>> model = Unilateral(graph_dict={
        ...     ("tumor", "T"): ["II", "III"],
        ...     ("lnl", "II"): ["III"],
        ...     ("lnl", "III"): [],
        ... })
        >>> model.set_params(0.7, 0.3, 0.2)  # doctest: +ELLIPSIS
        ()
        >>> model.transition_matrix()
        array([[0.21, 0.09, 0.49, 0.21],
               [0.  , 0.3 , 0.  , 0.7 ],
               [0.  , 0.  , 0.56, 0.44],
               [0.  , 0.  , 0.  , 1.  ]])

        """
        return matrix.generate_transition(
            lnls=self.graph.lnls.values(),
            num_states=3 if self.is_trinary else 2,
        )

    def observation_matrix(self) -> np.ndarray:
        r"""Get the matrix encoding the probabilities to observe a certain diagnosis.

        Every element in this matrix holds a probability to observe a certain diagnosis
        (or combination of diagnosis, when using multiple diagnostic modalities) given
        the current state of the system. It has the shape
        :math:`2^N \\times 2^\\{N \\times M\\}` where :math:`N` is the number of nodes
        in the graph and :math:`M` is the number of diagnostic modalities.

        .. seealso::

            :py:func:`~lymph.descriptors.matrix.generate_observation`
                The function actually computing the observation matrix.

        """
        return matrix.generate_observation(
            modalities=self.get_all_modalities().values(),
            num_lnls=len(self.graph.lnls),
            base=3 if self.is_trinary else 2,
        )

    def data_matrix(self, t_stage: str | None = None) -> np.ndarray:
        """Extract the data matrix for a given ``t_stage``.

        The data matrix is a binary encoding of the patient data. For every patient,
        it encodes the information which observational state could have led to the
        observed diagnosis. If a diagnosis is complete, i.e., for every diagnostic
        modality and every LNL we have an observation, the data matrix is a one-hot
        encoding of the observed diagnosis. Otherwise it may contain multiple 1s,
        indicating over which observational state one should marginalize.

        The data matrix is used to compute the :py:attr:`~diagnosis_matrix`, which in
        turn is used to compute the likelihood of the model given the patient data.

        .. seealso::

            :py:func:`.matrix.generate_data_encoding`
                This function actually computes the data encoding.

        """
        if self._patient_data is None:
            raise AttributeError("No patient data loaded yet.")

        # Compute entire data matrix if it is not in the cache
        full_hash = hash((None, self.modalities_hash(), self._cache_version))
        if full_hash not in self._data_matrix_cache:
            self._data_matrix_cache[full_hash] = matrix.generate_data_encoding(
                patient_data=self._patient_data,
                modalities=self.get_all_modalities(),
                lnls=list(self.graph.lnls.keys()),
            )

        # Extract a subset of the data matrix for a given T-stage. If `t_stage` is
        # `None`, this will be skipped and the entire data matrix will be returned.
        t_hash = hash((t_stage, self.modalities_hash(), self._cache_version))
        if t_hash not in self._data_matrix_cache:
            has_t_stage = self.patient_data[MAP_T_COL] == t_stage
            full_data_matrix = self._data_matrix_cache[full_hash]
            t_data_matrix = full_data_matrix[has_t_stage]
            self._data_matrix_cache[t_hash] = t_data_matrix

        return self._data_matrix_cache[t_hash]

    def diagnosis_matrix(self, t_stage: str | None = None) -> np.ndarray:
        """Extract the diagnosis matrix for a given ``t_stage``.

        For every patient this matrix stores the probability to observe this patient's
        diagnosis, given one of the possible hidden states of the model. It is computed
        by multiplying the :py:meth:`.data_matrix` with the
        :py:meth:`.observation_matrix`.
        """
        # Compute the entire diagnosis matrix if it is not in the cache. Note that this
        # requires the data matrix to be computed as well.
        _hash = hash((t_stage, self.modalities_hash(), self._cache_version))
        if _hash not in self._diagnosis_matrix_cache:
            self._diagnosis_matrix_cache[_hash] = (
                self.observation_matrix() @ self.data_matrix(t_stage).T
            )

        return self._diagnosis_matrix_cache[_hash].T

    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        side: str = "ipsi",
        mapping: Callable[[int], Any] | dict[int, Any] | None = None,
    ) -> None:
        """Load patient data in `LyProX`_ format into the model.

        Since the `LyProX`_ data format contains information on both sides (i.e.,
        ipsi- and contralateral) of the neck, the ``side`` parameter is used to select
        the for which of the two to store the involvement data.

        ``hpv_status`` is used to filter for patients with HPV status. If ``hpv_status``
        is set to ``True``, only patients with HPV status are kept. If ``hpv_status``
        is set to ``False``, only patients without HPV status are kept. If
        ``hpv_status`` is set to ``None``, all patients are kept.

        With the ``mapping`` function or dictionary, the reported T-stages (usually 0,
        1, 2, 3, and 4) can be mapped to any keys also used to access the corresponding
        distribution over diagnosis times. The default mapping is to map 0, 1, and 2 to
        "early" and 3 and 4 to "late".

        What this method essentially does is to copy the entire data frame, check all
        necessary information is present, and add a new top-level header ``"_model"`` to
        the data frame. Under this header, columns are assembled that contain all the
        information necessary to compute the observation and diagnosis matrices.

        .. _LyProX: https://lyprox.org/
        """
        if mapping is None:
            mapping = early_late_mapping

        patient_data = (
            patient_data.copy()
            .drop(columns="_model", errors="ignore")
            .reset_index(drop=True)
        )

        data_modalities = set(patient_data.columns.levels[0]) - {"patient", "tumor"}
        for modality in data_modalities:
            if side not in patient_data[modality]:
                warnings.warn(
                    f"{side}lateral involvement data not found. Skipping "
                    f"modality {modality}.",
                    category=types.DataWarning,
                )
                continue

            for lnl in self.graph.lnls.keys():
                modality_side_data = patient_data[modality, side]

                if lnl not in modality_side_data:
                    warnings.warn(
                        f"Modality {modality} does not contain involvement data for "
                        f"LNL {lnl}. Assuming unknown.",
                        category=types.DataWarning,
                    )
                    column = None
                else:
                    column = patient_data[modality, side, lnl]

                patient_data["_model", modality, lnl] = column

        patient_data[MAP_T_COL] = patient_data[RAW_T_COL].map(mapping)
        self._patient_data = patient_data
        self._cache_version += 1

        for t_stage in self.get_t_stages("distributions"):
            if t_stage not in patient_data[MAP_T_COL].values:
                warnings.warn(
                    message=f"No data for T-stage {t_stage} found.",
                    category=types.DataWarning,
                )

    @property
    def patient_data(self) -> pd.DataFrame:
        """Return the patient data loaded into the model.

        After successfully loading the data with :py:meth:`.load_patient_data`, the
        copied patient data now contains the additional top-level header ``"_model"``.
        Under it, the observed per LNL involvement is listed for every diagnostic
        modality in the dictionary returned by :py:meth:`.get_all_modalities` and for
        each of the LNLs in the list :py:attr:`.graph.Representation.lnls`.

        It also contains information on the patient's T-stage under the header
        ``("_model", "#", "t_stage")``.

        Additionally, it holds the data encodings and probability of diagnosis given the
        hidden states for each patient under the headers ``("_model", "_encoding",
        <obs_state>)`` and ``("_model", "_diagnosis_prob", <hidden_state>)``,
        respectively.
        """
        if self._patient_data is None:
            raise AttributeError("No patient data loaded yet.")

        # if not present, this will recompute the full data and diagnosis matrices
        _ = self.diagnosis_matrix()

        return self._patient_data

    def evolve(self, state_dist: np.ndarray, num_steps: int) -> np.ndarray:
        """Evolve the ``state_dist`` of possible states over ``num_steps``.

        This is done by multiplying the ``state_dist`` with the transition matrix
        from the left ``num_steps`` times. The result is a new distribution over
        possible states at a new time-step :math:`t' = t + n`, where :math:`n`
        is the number of steps ``num_steps``.
        """
        for _ in range(num_steps):
            state_dist = state_dist @ self.transition_matrix()

        return state_dist

    def state_dist_evo(self) -> np.ndarray:
        """Compute an evolution of the model's state distribution over time steps.

        This returns a matrix with the distribution over the possible states for
        each time step from :math:`t = 0` to :math:`t = T`, where :math:`T` is the
        maximum diagnosis time stored in the model's attribute ``max_time``.

        Note that at this point, the distributions are not weighted with the
        distribution over diagnosis times that are stored and managed for each T-stage
        in the dictionary returned by :py:meth:`.get_all_distributions`.
        """
        state_dists = np.zeros(shape=(self.max_time + 1, len(self.graph.state_list)))
        state_dists[0, 0] = 1.0

        for t in range(1, self.max_time + 1):
            state_dists[t] = self.evolve(state_dists[t - 1], num_steps=1)

        return state_dists

    def state_dist(
        self,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Compute the distribution over possible states.

        Do this either for a given ``t_stage``, when ``mode`` is set to ``"HMM"``,
        which is essentially a marginalization of the evolution over the possible
        states as computed by :py:meth:`.state_dist_evo` with the distribution
        over diagnosis times for the given T-stage from the dictionary returned by
        :py:meth:`.get_all_distributions`.

        Or, when ``mode`` is set to ``"BN"``, compute the distribution over states for
        the Bayesian network. In that case, the ``t_stage`` parameter is ignored.
        """
        if mode == "HMM":
            state_dists = self.state_dist_evo()
            diag_time_dist = self.get_distribution(t_stage).pmf

            return diag_time_dist @ state_dists

        if mode == "BN":
            state_dist = np.ones(shape=(len(self.graph.state_list),), dtype=float)

            for i, state in enumerate(self.graph.state_list):
                self.graph.set_state(*state)
                for node in self.graph.lnls.values():
                    state_dist[i] *= node.comp_bayes_net_prob()

            return state_dist

        raise ValueError("Invalid mode. Must be either 'HMM' or 'BN'.")

    def obs_dist(
        self,
        given_state_dist: np.ndarray | None = None,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Compute the distribution over all possible observations for a given T-stage.

        Returns an array of probabilities for each possible complete observation. This
        entails multiplying the distribution over states as returned by the
        :py:meth:`.state_dist` method with the :py:meth:`.observation_matrix`.

        Note that since the :py:attr:`.observation_matrix` can become very large, this
        method is not very efficient for inference. Instead, we compute the
        :py:meth:`.diagnosis_matrix` from the :py:meth:`.observation_matrix` and
        the :py:meth:`.data_matrix` and use these to compute the likelihood.
        """
        if given_state_dist is None:
            given_state_dist = self.state_dist(t_stage=t_stage, mode=mode)

        return given_state_dist @ self.observation_matrix()

    def _bn_likelihood(self, log: bool = True, t_stage: str | None = None) -> float:
        """Compute the BN likelihood, using the stored params."""
        state_dist = self.state_dist(mode="BN")
        patient_llhs = state_dist @ self.diagnosis_matrix(t_stage).T

        return np.sum(np.log(patient_llhs)) if log else np.prod(patient_llhs)

    def _hmm_likelihood(self, log: bool = True, t_stage: str | None = None) -> float:
        """Compute the HMM likelihood, using the stored params."""
        evolved_model = self.state_dist_evo()
        llh = 0.0 if log else 1.0

        if t_stage is None:
            t_stages = self.get_t_stages("valid")
        else:
            t_stages = [t_stage]

        for t_stage in t_stages:
            patient_llhs = (
                self.get_distribution(t_stage).pmf
                @ evolved_model
                @ self.diagnosis_matrix(t_stage).T
            )
            llh = add_or_mult(llh, patient_llhs, log)

        return llh

    def likelihood(
        self,
        given_params: types.ParamsType | None = None,
        log: bool = True,
        t_stage: str | None = None,
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> float:
        """Compute the (log-)likelihood of the stored data given the model (and params).

        See the documentation of :py:meth:`lymph.types.Model.likelihood` for more
        information on how to use the ``given_params`` parameter.

        Returns the log-likelihood if ``log`` is set to ``True``. The ``mode`` parameter
        determines whether the likelihood is computed for the hidden Markov model
        (``"HMM"``) or the Bayesian network (``"BN"``).
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

    def compute_encoding(
        self,
        given_diagnosis: types.DiagnosisType | None = None,
    ) -> np.ndarray:
        """Compute one-hot vector encoding of a given diagnosis."""
        diagnosis_encoding = np.array([True], dtype=bool)

        for modality in self.get_all_modalities().keys():
            diagnosis_encoding = np.kron(
                diagnosis_encoding,
                matrix.compute_encoding(
                    lnls=self.graph.lnls.keys(),
                    pattern=given_diagnosis.get(modality, {}),
                    base=2,  # diagnosis are always binary!
                ),
            )

        return diagnosis_encoding

    def posterior_state_dist(
        self,
        given_params: types.ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnosis: types.DiagnosisType | None = None,
        t_stage: str | int = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Compute the posterior distribution over hidden states given a diagnosis.

        The ``given_diagnosis`` is a dictionary of diagnosis for each modality. E.g.,
        this could look like this:

        .. code-block:: python

            given_diagnosis = {
                "MRI": {"II": True, "III": False, "IV": False},
                "PET": {"II": True, "III": True, "IV": None},
            }

        The ``t_stage`` parameter determines the T-stage for which the posterior is
        computed. The ``mode`` parameter determines whether the posterior is computed
        for the hidden Markov model (``"HMM"``) or the Bayesian network (``"BN"``).
        In case of the Bayesian network mode, the ``t_stage`` parameter is ignored.

        .. warning::

            To speed up repetitive computations, one can provide precomputed state
            distributions via the ``given_state_dist`` parameter. When provided, the
            method will ignore the ``given_params``, ``t_stage``, and ``mode``
            arguments, but compute the posterior much quicker.

        """
        if given_state_dist is None:
            # in contrast to when computing the likelihood, we do want to raise an error
            # here if the parameters are invalid, since we want to know if the user
            # provided invalid parameters.
            utils.safe_set_params(self, given_params)
            # vector P(X=x) of probs of arriving in state x (marginalized over time)
            given_state_dist = self.state_dist(t_stage, mode=mode)

        if given_diagnosis is None:
            return given_state_dist

        diagnosis_encoding = self.compute_encoding(given_diagnosis)
        # vector containing P(Z=z|X). Essentially a data matrix for one patient
        diagnosis_given_state = diagnosis_encoding @ self.observation_matrix().T

        # multiply P(Z=z|X) * P(X) element-wise to get vector of joint probs P(Z=z,X)
        joint_diagnosis_and_state = given_state_dist * diagnosis_given_state

        # compute vector of probabilities for all possible involvements given the
        # specified diagnosis P(X|Z=z) = P(Z=z,X) / P(X), where P(X) = sum_z P(Z=z,X)
        return joint_diagnosis_and_state / np.sum(joint_diagnosis_and_state)

    def marginalize(
        self,
        involvement: types.PatternType,
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

        marginalize_over_states = matrix.compute_encoding(
            lnls=self.graph.lnls.keys(),
            pattern=involvement,
            base=3 if self.is_trinary else 2,
        )
        return marginalize_over_states @ given_state_dist

    def risk(
        self,
        involvement: types.PatternType,
        given_params: types.ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnosis: dict[str, types.PatternType] | None = None,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> float:
        """Compute risk of a certain ``involvement``, using the ``given_diagnosis``.

        If an ``involvement`` pattern of interest is provided, this method computes
        the risk of seeing just that pattern for the set of given parameters and a
        dictionary of diagnosis for each modality.

        If no ``involvement`` is provided, this will simply return the posterior
        distribution over hidden states, given the diagnosis, as computed by the
        :py:meth:`.posterior_state_dist` method. See its documentation for more
        details about the arguments and the return value.
        """
        posterior_state_dist = self.posterior_state_dist(
            given_params=given_params,
            given_state_dist=given_state_dist,
            given_diagnosis=given_diagnosis,
            t_stage=t_stage,
            mode=mode,
        )

        # if a specific involvement of interest is provided, marginalize the
        # resulting vector of hidden states to match that involvement of
        # interest
        return self.marginalize(involvement, posterior_state_dist)

    def draw_diagnosis(  # noqa: F811
        self,
        diag_times: list[int],
        rng: np.random.Generator | None = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Given some ``diag_times``, draw diagnosis for each LNL.

        >>> model = Unilateral(graph_dict={
        ...     ("tumor", "T"): ["II" , "III"],
        ...     ("lnl", "II"): ["III"],
        ...     ("lnl", "III"): [],
        ... })
        >>> model.set_modality("CT", spec=0.8, sens=0.8)
        >>> model.draw_diagnosis([0, 1, 2, 3, 4])       # doctest: +NORMALIZE_WHITESPACE
        array([[False,  True],
               [False, False],
               [ True, False],
               [False,  True],
               [False, False]])
        >>> draw_diagnosis(                   # this is the same as the previous example
        ...     diagnosis_times=[0, 1, 2, 3, 4],
        ...     state_evolution=model.state_dist_evo(),
        ...     observation_matrix=model.observation_matrix(),
        ...     possible_diagnosis=model.obs_list,
        ... )
        array([[False,  True],
               [False, False],
               [ True, False],
               [False,  True],
               [False, False]])
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        state_probs_given_time = self.state_dist_evo()[diag_times]
        obs_probs_given_time = state_probs_given_time @ self.observation_matrix()

        obs_indices = np.arange(len(self.obs_list))
        drawn_obs_idx = [
            rng.choice(obs_indices, p=obs_prob) for obs_prob in obs_probs_given_time
        ]

        return self.obs_list[drawn_obs_idx].astype(bool)

    def draw_patients(
        self,
        num: int,
        stage_dist: Iterable[float],
        rng: np.random.Generator | None = None,
        seed: int = 42,
        **_kwargs,
    ) -> pd.DataFrame:
        """Draw ``num`` random patients from the model.

        For this, a ``stage_dist``, i.e., a distribution over the T-stages, needs to
        be defined. This must be an iterable of probabilities with as many elements as
        there are defined T-stages in the model (accessible via
        :py:meth:`.get_all_distributions`).

        A random number generator can be provided as ``rng``. If ``None``, a new one
        is initialized with the given ``seed`` (or ``42``, by default).

        .. seealso::

            :py:meth:`lymph.diagnosis_times.Distribution.draw_diag_times`
                Method to draw diagnosis times from a distribution.
            :py:meth:`lymph.models.Unilateral.draw_diagnosis`
                Method to draw individual diagnosis.
            :py:meth:`lymph.models.Bilateral.draw_patients`
                The corresponding bilateral method.
        """
        rng = rng or np.random.default_rng(seed)

        if sum(stage_dist) != 1.0:
            warnings.warn("Sum of stage distribution is not 1. Renormalizing.")
            stage_dist = np.array(stage_dist) / sum(stage_dist)

        drawn_t_stages = rng.choice(
            a=self.get_t_stages("distributions"),
            p=stage_dist,
            size=num,
        )
        distributions = self.get_all_distributions()
        drawn_diag_times = [
            distributions[t_stage].draw_diag_times(rng=rng)
            for t_stage in drawn_t_stages
        ]

        drawn_obs = self.draw_diagnosis(drawn_diag_times, rng=rng)

        modality_names = list(self.get_all_modalities().keys())
        lnl_names = list(self.graph.lnls.keys())
        multi_cols = pd.MultiIndex.from_product([modality_names, ["ipsi"], lnl_names])

        dataset = pd.DataFrame(drawn_obs, columns=multi_cols)
        dataset[(RAW_T_COL)] = drawn_t_stages

        return dataset
