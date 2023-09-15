from __future__ import annotations

import itertools
import warnings
from itertools import product
from typing import Any, Callable, Generator, Iterable, Iterator

import numpy as np
import pandas as pd

from lymph import diagnose_times, graph, matrix, modalities
from lymph.helper import (
    DelegatorMixin,
    DiagnoseType,
    PatternType,
    early_late_mapping,
    not_updateable_cached_property,
    smart_updating_dict_cached_property,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class Unilateral(DelegatorMixin):
    """Class that models metastatic progression in a unilateral lymphatic system.

    It does this by representing it as a directed graph (DAG), which is stored in and
    managed by the attribute :py:attr:`~graph`. The progression itself can be modelled
    via hidden Markov models (HMM) or Bayesian networks (BN). In both cases, instances
    of this class allow to calculate the probability of a certain hidden pattern of
    involvement, given an individual diagnosis of a patient.
    """
    is_binary: bool
    is_trinary: bool
    get_state: Callable
    set_state: Callable
    state_list: list[int]
    lnls: dict[str, graph.LymphNodeLevel]

    def __init__(
        self,
        graph_dict: dict[tuple[str], list[str]],
        tumor_state: int | None = None,
        allowed_states: list[int] | None = None,
        max_time: int = 10,
        is_micro_mod_shared: bool = False,
        is_growth_shared: bool = False,
        **_kwargs,
    ) -> None:
        """Create a new instance of the :py:class:`~Unilateral` class.

        The ``graph_dict`` that represents the lymphatic system should given as a
        dictionary. Its keys are tuples of the form ``("tumor", "<tumor_name>")`` or
        ``("lnl", "<lnl_name>")``. The values are lists of strings that represent the
        names of the nodes that are connected to the node given by the key.

        Note:
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
        diagnosis. In the HMM case, the probability disitrubtion over all hidden states
        is evolved from :math:`t=0` to ``max_time``. In the BN case, this parameter has
        no effect.

        The ``is_micro_mod_shared`` and ``is_growth_shared`` parameters determine
        whether the microscopic involvement and growth parameters are shared among all
        LNLs. If they are set to ``True``, the parameters are set globally for all LNLs.
        If they are set to ``False``, the parameters are set individually for each LNL.
        """
        super().__init__()

        self.graph = graph.Representation(
            graph_dict=graph_dict,
            tumor_state=tumor_state,
            allowed_states=allowed_states,
            on_edge_change=[self.delete_transition_matrix],
        )

        if 0 >= max_time:
            raise ValueError("Latest diagnosis time `max_time` must be positive int")

        self.max_time = max_time
        self.is_micro_mod_shared = is_micro_mod_shared
        self.is_growth_shared = is_growth_shared

        self.init_delegation(
            graph=[
                "is_binary", "is_trinary",
                "get_state", "set_state", "state_list",
                "lnls",
            ],
        )


    @classmethod
    def binary(cls, graph_dict: dict[tuple[str], set[str]], **kwargs) -> Unilateral:
        """Create an instance of the :py:class:`~Unilateral` class with binary LNLs."""
        return cls(graph_dict, allowed_states=[0, 1], **kwargs)


    @classmethod
    def trinary(cls, graph_dict: dict[tuple[str], set[str]], **kwargs) -> Unilateral:
        """Create an instance of the :py:class:`~Unilateral` class with trinary LNLs."""
        return cls(graph_dict, allowed_states=[0, 1, 2], **kwargs)


    def __str__(self) -> str:
        """Print info about the instance."""
        return f"Unilateral with {len(self.graph.tumors)} tumors and {len(self.graph.lnls)} LNLs"


    def print_info(self):
        """Print detailed information about the instance."""
        num_tumors = len(self.graph.tumors)
        num_lnls   = len(self.graph.lnls)
        string = (
            f"Unilateral lymphatic system with {num_tumors} tumor(s) "
            f"and {num_lnls} LNL(s).\n"
            + " ".join([f"{e} {e.spread_prob}%" for e in self.graph.tumor_edges]) + "\n" + " ".join([f"{e} {e.spread_prob}%" for e in self.graph.lnl_edges])
            + f"\n the growth probability is: {self.graph.growth_edges[0].spread_prob}" + f" the micro mod is {self.graph.lnl_edges[0].micro_mod}"
        )
        print(string)


    def get_params(
        self,
        param: str | None = None,
        as_dict: bool = False,
        with_edges: bool = True,
        with_dists: bool = True,
    ) -> float | Iterable[float] | dict[str, float]:
        """Get the parameters of the model.

        If ``as_dict`` is ``True``, return a dictionary with the parameters as values.
        Otherwise, return the value of the parameter ``param``.

        Using the keyword arguments ``with_edges`` and ``with_dists``, one can control
        whether the parameters of the edges and the distributions over diagnose times
        should be included in the returned parameters. By default, both are included.
        """
        iterator = []
        params = {}

        if with_edges:
            iterator = itertools.chain(iterator, self.graph.edges.items())

        if with_dists:
            iterator = itertools.chain(iterator, self.diag_time_dists.items())

        for edge_name_or_tstage, edge_or_dist in iterator:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                edge_or_dist_params = edge_or_dist.get_params(as_dict=True)
            for name, value in edge_or_dist_params.items():
                params[f"{edge_name_or_tstage}_{name}"] = value

        if param is not None:
            return params[param]

        return params if as_dict else params.values()


    def _assign_via_args(self, new_params_args: Iterator[float]) -> Iterator[float]:
        """Assign parameters to egdes and to distributions via positional arguments."""
        for edge_or_dist in itertools.chain(
            self.graph.edges.values(),
            self.diag_time_dists.values(),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                params = edge_or_dist.get_params(as_dict=True)

            new_params = {}
            for name in params:
                try:
                    new_params[name] = next(new_params_args)
                except StopIteration:
                    return new_params_args
                finally:
                    edge_or_dist.set_params(**new_params)

        return new_params_args


    def _assign_via_kwargs(
        self,
        new_params_kwargs: dict[str, float],
    ) -> dict[str, float]:
        """Assign parameters to egdes and to distributions via keyword arguments."""
        remaining_kwargs = {}

        global_growth_param = new_params_kwargs.pop("growth", None)
        if self.is_growth_shared and global_growth_param is not None:
            for growth_edge in self.graph.growth_edges.values():
                growth_edge.set_spread_prob(global_growth_param)

        global_micro_mod = new_params_kwargs.pop("micro", None)
        if self.is_micro_mod_shared and global_micro_mod is not None:
            for lnl_edge in self.graph.lnl_edges.values():
                lnl_edge.set_micro_mod(global_micro_mod)

        edges_and_dists = self.graph.edges.copy()
        edges_and_dists.update(self.diag_time_dists)
        for key, value in new_params_kwargs.items():
            edge_name_or_tstage, type_ = key.rsplit("_", maxsplit=1)
            if edge_name_or_tstage in edges_and_dists:
                edge_or_dist = edges_and_dists[edge_name_or_tstage]
                edge_or_dist.set_params(**{type_: value})

        return remaining_kwargs


    def assign_params(
        self,
        *new_params_args,
        **new_params_kwargs,
    ) -> tuple[Iterator[float], dict[str, float]]:
        """Assign new parameters to the model.

        The parameters can either be provided with positional arguments or as
        keyword arguments. The positional arguments must be in the following order:

        1. All spread probs from tumor to the LNLs
        2. The parameters of arcs from LNL to LNL. For each arc, the parameters are set
           in the following order:

            1. The spread probability (or growth probability, if it's a growth edge)
            2. The microscopic involvement probability, if the model is trinary

        3. The parameters for the marginalizing distributions over diagnose times. Note
           that a distribution may take more than one parameter. So, if there are e.g.
           two T-stages with distributions over diagnose times that take two parameters
           each, this step requires and consumes four arguments.

        If the arguments are not used up, the remaining ones are given back as the first
        element of the returned tuple.

        When providing keyword arguments, the order of the keyword arguments obviously
        does not matter. If one wants to set the microscopic or growth parameters
        globally for all LNLs, the keyword arguments ``micro`` and ``growth`` can
        be used for that.

        As with the positional arguments, the dictionary of unused keyword arguments is
        returned as the second element of the tuple.

        Note:
            Providing positional arguments does not allow using the global
            parameters ``micro`` and ``growth``.

            However, when assigning them via keyword arguments, the global parameters
            are set first, while still allowing to override them for individual edges.

        Since the distributions over diagnose times may take more than one parameter,
        they can be provided as keyword arguments by appending their name to the
        corresponding T-stage, separated by an underscore. For example, a parameter
        ``foo`` for the T-stage ``early`` is set via the keyword argument ``early_foo``.

        Note:
            When using keyword arguments to set the parameters of the distributions
            over diagnose times, it is not possible to just use the name of the
            T-stage, even when the distribution only takes one parameter.

        The keyword arguments override the positional arguments, when both are provided.

        Example:

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
        >>> _ = model.assign_params(
        ...     0.7, 0.5, 0.3, 0.2, 0.1, 0.4
        ... )
        >>> model.get_params(as_dict=True)  # doctest: +NORMALIZE_WHITESPACE
        {'T_to_II_spread': 0.7,
         'T_to_III_spread': 0.5,
         'II_growth': 0.3,
         'II_to_III_spread': 0.2,
         'II_to_III_micro': 0.1,
         'III_growth': 0.4}
        >>> _ = model.assign_params(growth=0.123)
        >>> model.get_params(as_dict=True)  # doctest: +NORMALIZE_WHITESPACE
        {'T_to_II_spread': 0.7,
         'T_to_III_spread': 0.5,
         'II_growth': 0.123,
         'II_to_III_spread': 0.2,
         'II_to_III_micro': 0.1,
         'III_growth': 0.123}
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            remaining_args = self._assign_via_args(iter(new_params_args))
            remainig_kwargs = self._assign_via_kwargs(new_params_kwargs)

        return remaining_args, remainig_kwargs


    def comp_transition_prob(
        self,
        newstate: list[int],
        assign: bool = False
    ) -> float:
        """Computes probability to transition to ``newstate``, given its current state.

        The probability is computed as the product of the transition probabilities of
        the individual LNLs. If ``assign`` is ``True``, the new state is assigned to
        the model using the method :py:meth:`~Unilateral.assign_states`.
        """
        trans_prob = 1
        for i, lnl in enumerate(self.graph.lnls):
            trans_prob *= lnl.comp_trans_prob(new_state = newstate[i])
            if trans_prob == 0:
                break

        if assign:
            self.graph.set_state(newstate)

        return trans_prob


    def comp_diagnose_prob(
        self,
        diagnoses: pd.Series | dict[str, dict[str, bool]]
    ) -> float:
        """Compute the probability to observe a diagnose given the current state.

        The ``diagnoses`` is either a pandas ``Series`` object corresponding to one row
        of a patient data table, or a dictionary with keys of diagnostic modalities and
        values of dictionaries holding the observation for each LNL under the
        respective key.

        It returns the probability of observing this particular combination of
        diagnoses, given the current state of the system.
        """
        prob = 1.
        for name, modality in self.modalities.items():
            if name in diagnoses:
                mod_diagnose = diagnoses[name]
                for lnl in self.graph.lnls:
                    try:
                        lnl_diagnose = mod_diagnose[lnl.name]
                    except KeyError:
                        continue
                    except IndexError as idx_err:
                        raise ValueError(
                            "diagnoses were not provided in the correct format"
                        ) from idx_err
                    prob *= lnl.comp_obs_prob(lnl_diagnose, modality.confusion_matrix)
        return prob


    def _gen_obs_list(self):
        """Generates the list of possible observations."""
        possible_obs_list = []
        for modality in self.modalities.values():
            possible_obs = np.arange(modality.confusion_matrix.shape[1])
            for _ in self.graph.lnls:
                possible_obs_list.append(possible_obs.copy())

        self._obs_list = np.array(list(product(*possible_obs_list)))

    @property
    def obs_list(self):
        """Return the list of all possible observations.

        They are ordered the same way as the :py:attr:`~Unilateral.state_list`, but
        additionally by modality. E.g., for two LNLs II, III and two modalities CT,
        pathology, the list would look like this:

        >>> model = Unilateral(graph_dict={
        ...     ("tumor", "T"): ["II" , "III"],
        ...     ("lnl", "II"): ["III"],
        ...     ("lnl", "III"): [],
        ... })
        >>> model.modalities = {
        ...     "CT":        (0.8, 0.8),
        ...     "pathology": (1.0, 1.0),
        ... }
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
        try:
            return self._obs_list
        except AttributeError:
            self._gen_obs_list()
            return self._obs_list

    @obs_list.deleter
    def obs_list(self):
        """Delete the observation list. Necessary to pass as callback."""
        if hasattr(self, "_obs_list"):
            del self._obs_list


    @not_updateable_cached_property
    def transition_matrix(self) -> np.ndarray:
        """Matrix encoding the probabilities to transition from one state to another.

        This is the crucial object for modelling the evolution of the probabilistic
        system in the context of the hidden Markov model. It has the shape
        :math:`2^N \\times 2^N` where :math:`N` is the number of nodes in the graph.
        The :math:`i`-th row and :math:`j`-th column encodes the probability to
        transition from the :math:`i`-th state to the :math:`j`-th state. The states
        are ordered as in the :py:attr:`lymph.graph.state_list`.

        This matrix is deleted every time the parameters along the edges of the graph
        are changed. It is lazily computed when it is next accessed.

        See Also:
            :py:func:`~lymph.descriptors.matrix.generate_transition`
                The function actually computing the transition matrix.

        Example:

        >>> model = Unilateral(graph_dict={
        ...     ("tumor", "T"): ["II", "III"],
        ...     ("lnl", "II"): ["III"],
        ...     ("lnl", "III"): [],
        ... })
        >>> model.assign_params(0.7, 0.3, 0.2)
        >>> model.transition_matrix
        array([[0.21, 0.09, 0.49, 0.21],
            [0.  , 0.3 , 0.  , 0.7 ],
            [0.  , 0.  , 0.56, 0.44],
            [0.  , 0.  , 0.  , 1.  ]])
        """
        return matrix.generate_transition(self)

    def delete_transition_matrix(self):
        """Delete the transition matrix. Necessary to pass as callback."""
        del self.transition_matrix


    @smart_updating_dict_cached_property
    def modalities(self) -> modalities.ModalitiesUserDict:
        """Dictionary of diagnostic modalities and their confusion matrices.

        This must be set by the user. For example, if one wanted to add the modality
        "CT" with a sensitivity of 80% and a specificity of 90%, one would do:

        >>> model = Unilateral(graph_dict={
        ...     ("tumor", "T"): ["II", "III"],
        ...     ("lnl", "II"): ["III"],
        ...     ("lnl", "III"): [],
        ... })
        >>> model.modalities["CT"] = (0.8, 0.9)

        See Also:
            :py:class:`~lymph.descriptors.modalities.ModalitiesUserDict`
            :py:class:`~lymph.descriptors.modalities.Modality`
        """
        return modalities.ModalitiesUserDict(
            is_trinary=self.is_trinary,
            trigger_callbacks=[self.delete_obs_list_and_matrix],
        )


    @not_updateable_cached_property
    def observation_matrix(self) -> np.ndarray:
        """The matrix encoding the probabilities to observe a certain diagnosis.

        Every element in this matrix holds a probability to observe a certain diagnosis
        (or combination of diagnoses, when using multiple diagnostic modalities) given
        the current state of the system. It has the shape
        :math:`2^N \\times 2^\\{N \\times M\\}` where :math:`N` is the number of nodes in
        the graph and :math:`M` is the number of diagnostic modalities.

        See Also:
            :py:func:`~lymph.descriptors.matrix.generate_observation`
                The function actually computing the observation matrix.
        """
        return matrix.generate_observation(self)

    def delete_obs_list_and_matrix(self):
        """Delete the observation matrix. Necessary to pass as callback."""
        del self.observation_matrix
        del self.obs_list


    @smart_updating_dict_cached_property
    def data_matrices(self) -> matrix.DataEncodingUserDict:
        """Holds the data encoding in matrix form for every T-stage.

        See Also:
            :py:class:`~lymph.descriptors.matrix.DataEncodingUserDict`
        """
        return matrix.DataEncodingUserDict(model=self)


    @smart_updating_dict_cached_property
    def diagnose_matrices(self) -> matrix.DiagnoseUserDict:
        """Holds the probability of a patient's diagnosis, given any hidden state.

        Essentially, this is just the data encoding matrix of a certain T-stage
        multiplied with the observation matrix. It is thus also a dictionary with
        keys of T-stages and values of matrices.

        See Also:
            :py:class:`~lymph.descriptors.matrix.DiagnoseUserDict`
        """
        return matrix.DiagnoseUserDict(model=self)


    @smart_updating_dict_cached_property
    def diag_time_dists(self) -> diagnose_times.DistributionsUserDict:
        """Dictionary of distributions over diagnose times for each T-stage."""
        return diagnose_times.DistributionsUserDict(max_time=self.max_time)


    @property
    def t_stages(self) -> Generator[str, None, None]:
        """Generator of all valid T-stages in the model.

        This is the intersection of the unique T-stages found in the (mapped) data
        and the T-stages defined in the distributions over diagnose times.
        """
        for t_stage in self.diag_time_dists.keys():
            # This implementation is a little special, because the diagnose matrix
            # of a particular T-stage is only computed when either __contains__ or
            # __getitem__ is called on it. Therefore, we cannot directly loop over
            # the diagnose matrices' keys or something like that.
            if t_stage in self.diagnose_matrices:
                yield t_stage


    @property
    def stacked_diagnose_matrix(self) -> np.ndarray:
        """Stacked version of all T-stage's :py:attr:`~diagnose_matrices`.

        This is mainly used for the Bayesian network implementation of the model, which
        cannot naturally incorporate the T-stage as a random variable.
        """
        return np.hstack(list(self.diagnose_matrices.values()))


    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        side: str = "ipsi",
        mapping: callable = early_late_mapping,
    ) -> None:
        """Load patient data in `LyProX`_ format into the model.

        Since the `LyProX`_ data format contains information on both sides (i.e.,
        ipsi- and contralateral) of the neck, the ``side`` parameter is used to select
        the for which of the two to store the involvement data.

        With the ``mapping`` function, the reported T-stages (usually 0, 1, 2, 3, and 4)
        can be mapped to any keys also used to access the corresponding distribution
        over diagnose times. The default mapping is to map 0, 1, and 2 to "early" and
        3 and 4 to "late".

        What this method essentially does is to copy the entire data frame, check all
        necessary information is present, and add a new top-level header ``"_model"`` to
        the data frame. Under this header, columns are assembled that contain all the
        information necessary to compute the observation and diagnose matrices.

        .. _LyProX: https://lyprox.org/
        """
        patient_data = patient_data.copy()

        if mapping is None:
            mapping = {"early": [0,1,2], "late": [3,4]}

        for modality_name in self.modalities.keys():
            if modality_name not in patient_data:
                raise ValueError(f"Modality '{modality_name}' not found in data.")

            if side not in patient_data[modality_name]:
                raise ValueError(f"{side}lateral involvement data not found.")

            for name in self.graph.lnls:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                    modality_side_data = patient_data[modality_name, side]

                if name not in modality_side_data:
                    raise ValueError(f"Involvement data for LNL {name} not found.")
                column = patient_data[modality_name, side, name]
                patient_data["_model", modality_name, name] = column

        patient_data["_model", "#", "t_stage"] = patient_data.apply(
            lambda row: mapping(row["tumor", "1", "t_stage"]), axis=1
        )

        for t_stage in self.diag_time_dists.keys():
            if t_stage not in patient_data["_model", "#", "t_stage"].values:
                warnings.warn(f"No data for T-stage {t_stage} found.")

        # Changes to the patient data require a recomputation of the data and
        # diagnose matrices. Clearing them will trigger this when they are next
        # accessed.
        self.data_matrices.clear()
        self.diagnose_matrices.clear()
        self._patient_data = patient_data


    @property
    def patient_data(self) -> pd.DataFrame:
        """Return the patient data loaded into the model.

        After succesfully loading the data with the method :py:meth:`load_patient_data`,
        the copied patient data now contains the additional top-level header
        ``"_model"``. Under it, the observed per LNL involvement is listed for every
        diagnostic modality in the dictionary :py:attr:`~modalities` and for each of
        the LNLs in the list :py:attr:`~lnls`.

        It also contains information on the patient's T-stage under the header
        ``("_model", "#", "t_stage")``.
        """
        if not hasattr(self, "_patient_data"):
            raise AttributeError("No patient data loaded yet.")

        return self._patient_data


    def evolve_dist(self, state_dist: np.ndarray, num_steps: int) -> np.ndarray:
        """Evolve the ``state_dist`` of possible states over ``num_steps``.

        This is done by multiplying the ``state_dist`` with the transition matrix
        from the left ``num_steps`` times. The result is a new distribution over
        possible states at a new time-step :math:`t' = t + n`, where :math:`n`
        is the number of steps ``num_steps``.
        """
        for _ in range(num_steps):
            state_dist = state_dist @ self.transition_matrix

        return state_dist


    def comp_dist_evolution(self) -> np.ndarray:
        """Compute a complete evolution of the model.

        This returns a matrix with the distribution over the possible states for
        each time step from :math:`t = 0` to :math:`t = T`, where :math:`T` is the
        maximum diagnose time stored in the model's attribute ``max_time``.

        Note that at this point, the distributions are not weighted with the
        distribution over diagnose times that are stored and managed for each T-stage
        in the dictionary :py:attr:`~diag_time_dists`.
        """
        state_dists = np.zeros(shape=(self.max_time + 1, len(self.state_list)))
        state_dists[0, 0] = 1.

        for t in range(1, self.max_time + 1):
            state_dists[t] = self.evolve_dist(state_dists[t-1], num_steps=1)

        return state_dists


    def comp_state_dist(self, t_stage: str = "early", mode: str = "HMM") -> np.ndarray:
        """Compute the distribution over possible states.

        Do this either for a given ``t_stage``, when ``mode`` is set to ``"HMM"``,
        which is essentially a marginalization of the evolution over the possible
        states as computed by :py:meth:`~comp_dist_evolution` with the distribution
        over diagnose times for the given T-stage from the dictionary
        :py:attr:`~diag_time_dists`.

        Or, when ``mode`` is set to ``"BN"``, compute the distribution over states for
        the Bayesian network. In that case, the ``t_stage`` parameter is ignored.
        """
        if mode == "HMM":
            state_dists = self.comp_dist_evolution()
            diag_time_dist = self.diag_time_dists[t_stage].distribution

            return diag_time_dist @ state_dists

        if mode == "BN":
            state_dist = np.ones(shape=(len(self.state_list),), dtype=float)

            for i, state in enumerate(self.state_list):
                self.set_state(*state)
                for node in self.graph.lnls.values():
                    state_dist[i] *= node.comp_bayes_net_prob()

            return state_dist


    def comp_obs_dist(self, t_stage: str = "early", mode: str = "HMM") -> np.ndarray:
        """Compute the distribution over all possible observations for a given T-stage.

        Returns an array of probabilities for each possible complete observation. This
        entails multiplying the distribution over states as returned by the
        :py:meth:`~comp_state_dist` method with the :py:attr:`~observation_matrix`.

        Note that since the :py:attr:`~observation_matrix` can become very large, this
        method is not very efficient for inference. Instead, we compute the
        :py:attr:`~diagnose_matrices` from the :py:attr:`~observation_matrix` and
        the :py:attr:`~data_matrices` and use these to compute the likelihood.
        """
        state_dist = self.comp_state_dist(t_stage=t_stage, mode=mode)
        return state_dist @ self.observation_matrix


    def _bn_likelihood(self, log: bool = True) -> float:
        """Compute the BN likelihood, using the stored params."""
        state_dist = self.comp_state_dist(mode="BN")
        patient_likelihoods = state_dist @ self.stacked_diagnose_matrix

        if log:
            llh = np.sum(np.log(patient_likelihoods))
        else:
            llh = np.prod(patient_likelihoods)
        return llh


    def _hmm_likelihood(self, log: bool = True) -> float:
        """Compute the HMM likelihood, using the stored params."""
        evolved_model = self.comp_dist_evolution()
        llh = 0. if log else 1.

        for t_stage in self.t_stages:
            patient_likelihoods = (
                self.diag_time_dists[t_stage].distribution
                @ evolved_model
                @ self.diagnose_matrices[t_stage]
            )
            if log:
                llh += np.sum(np.log(patient_likelihoods))
            else:
                llh *= np.prod(patient_likelihoods)

        return llh


    def likelihood(
        self,
        data: pd.DataFrame | None = None,
        given_param_args: Iterable[float] | None = None,
        given_param_kwargs: dict[str, float] | None = None,
        load_data_kwargs: dict[str, Any] | None = None,
        log: bool = True,
        mode: str = "HMM"
    ) -> float:
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


    def comp_diagnose_encoding(
        self,
        given_diagnoses: DiagnoseType | None = None,
    ) -> np.ndarray:
        """Compute one-hot vector encoding of a given diagnosis."""
        diagnose_encoding = np.array([True], dtype=bool)

        for modality in self.modalities.keys():
            diagnose_encoding = np.kron(
                diagnose_encoding,
                matrix.compute_encoding(
                    lnls=[lnl.name for lnl in self.graph.lnls],
                    pattern=given_diagnoses.get(modality, {}),
                ),
            )

        return diagnose_encoding


    def comp_posterior_state_dist(
        self,
        given_param_args: Iterable[float] | None = None,
        given_param_kwargs: dict[str, float] | None = None,
        given_diagnoses: DiagnoseType | None = None,
        t_stage: str | int = "early",
        mode: str = "HMM",
    ) -> np.ndarray:
        """Compute the posterior distribution over hidden states given a diagnosis.

        The ``given_diagnoses`` is a dictionary of diagnoses for each modality. E.g.,
        this could look like this:

        .. code-block:: python

            given_diagnoses = {
                "MRI": {"II": True, "III": False, "IV": False},
                "PET": {"II": True, "III": True, "IV": None},
            }

        The ``t_stage`` parameter determines the T-stage for which the posterior is
        computed. The ``mode`` parameter determines whether the posterior is computed
        for the hidden Markov model (``"HMM"``) or the Bayesian network (``"BN"``).
        In case of the Bayesian network mode, the ``t_stage`` parameter is ignored.

        Note:
            The computation is much faster if no parameters are given, since then the
            transition matrix does not need to be recomputed.
        """
        if given_param_args is None:
            given_param_args = []

        if given_param_kwargs is None:
            given_param_kwargs = {}

        # in contrast to when computing the likelihood, we do want to raise an error
        # here if the parameters are invalid, since we want to know if the user
        # provided invalid parameters. In the likelihood, we rather return a zero
        # likelihood to tell the inference algorithm that the parameters are invalid.
        self.assign_params(*given_param_args, **given_param_kwargs)

        if given_diagnoses is None:
            given_diagnoses = {}

        diagnose_encoding = self.comp_diagnose_encoding(given_diagnoses)
        # vector containing P(Z=z|X). Essentially a data matrix for one patient
        diagnose_given_state = diagnose_encoding @ self.observation_matrix

        # vector P(X=x) of probabilities of arriving in state x (marginalized over time)
        state_dist = self.comp_state_dist(t_stage, mode=mode)

        # multiply P(Z=z|X) * P(X) elementwise to get vector of joint probs P(Z=z,X)
        joint_diagnose_and_state = state_dist * diagnose_given_state

        # compute vector of probabilities for all possible involvements given the
        # specified diagnosis P(X|Z=z) = P(Z=z,X) / P(X), where P(X) = sum_z P(Z=z,X)
        return joint_diagnose_and_state / np.sum(joint_diagnose_and_state)


    def risk(
        self,
        involvement: PatternType | None = None,
        given_param_args: Iterable[float] | None = None,
        given_param_kwargs: dict[str, float] | None = None,
        given_diagnoses: dict[str, PatternType] | None = None,
        t_stage: str = "early",
        mode: str = "HMM",
        **_kwargs,
    ) -> float | np.ndarray:
        """Compute risk of a certain involvement, given a patient's diagnosis.

        If an ``involvement`` pattern of interest is provided, this method computes
        the risk of seeing just that pattern for the set of given parameters and a
        dictionary of diagnoses for each modality.

        Using the ``mode`` parameter, the risk can be computed either for the hidden
        Markov model (``"HMM"``) or the Bayesian network (``"BN"``). In case of the
        Bayesian network mode, the ``t_stage`` parameter is ignored.

        Note:
            The computation is much faster if no parameters are given, since then the
            transition matrix does not need to be recomputed.

        See Also:
            :py:meth:`comp_posterior_state_dist`
        """
        posterior_state_dist = self.comp_posterior_state_dist(
            given_param_args, given_param_kwargs, given_diagnoses, t_stage, mode,
        )

        if involvement is None:
            return posterior_state_dist

        # if a specific involvement of interest is provided, marginalize the
        # resulting vector of hidden states to match that involvement of
        # interest
        marginalize_over_states = matrix.compute_encoding(
            lnls=[lnl.name for lnl in self.graph.lnls],
            pattern=involvement,
        )
        return marginalize_over_states @ posterior_state_dist


    def _draw_patient_diagnoses(
        self,
        diag_times: list[int],
    ) -> np.ndarray:
        """Draw random possible observations for a list of T-stages and
        diagnose times.

        Args:
            diag_times: List of diagnose times for each patient who's diagnose
                is supposed to be drawn.
        """
        # use the drawn diagnose times to compute probabilities over states and
        # diagnoses
        per_time_state_probs = self.comp_dist_evolution()
        per_patient_state_probs = per_time_state_probs[diag_times]
        per_patient_obs_probs = per_patient_state_probs @ self.observation_matrix

        # then, draw a diagnose from the possible ones
        obs_idx = np.arange(len(self.obs_list))
        drawn_obs_idx = [
            np.random.choice(obs_idx, p=obs_prob)
            for obs_prob in per_patient_obs_probs
        ]
        return self.obs_list[drawn_obs_idx].astype(bool)


    def generate_dataset(
        self,
        num_patients: int,
        stage_dist: dict[str, float],
        **_kwargs,
    ) -> pd.DataFrame:
        """Generate/sample a pandas :class:`DataFrame` from the defined network
        using the samples and diagnostic modalities that have been set.

        Args:
            num_patients: Number of patients to generate.
            stage_dist: Probability to find a patient in a certain T-stage.
        """
        drawn_t_stages, drawn_diag_times = self.diag_time_dists.draw(
            prob_of_t_stage=stage_dist, size=num_patients
        )

        drawn_obs = self._draw_patient_diagnoses(drawn_diag_times)

        # construct MultiIndex for dataset from stored modalities
        modality_names = list(self.modalities.keys())
        lnl_names = [lnl.name for lnl in self.graph.lnls]
        multi_cols = pd.MultiIndex.from_product([modality_names, lnl_names])

        # create DataFrame
        dataset = pd.DataFrame(drawn_obs, columns=multi_cols)
        dataset[('info', 't_stage')] = drawn_t_stages

        return dataset
