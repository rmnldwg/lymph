from __future__ import annotations

import base64
import itertools
import warnings
from itertools import product

import numpy as np
import pandas as pd

from lymph import graph
from lymph.descriptors import diagnose_times, matrix, modalities, params
from lymph.helper import PatternType, early_late_mapping

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class Unilateral:
    """
    Class that models metastatic progression in a (unilateral) lymphatic system.

    It does this by representing it as a directed graph (DAG). The progression itself
    can be modelled via hidden Markov models (HMM) or Bayesian networks (BN). In both
    cases, instances of this class allow to calculate the probability of a certain
    hidden pattern of involvement, given an individual diagnosis of a patient.
    """
    def __init__(
        self,
        graph_dict: dict[tuple[str], list[str]],
        tumor_state: int | None = None,
        allowed_states: list[int] | None = None,
        max_time: int = 10,
        **_kwargs,
    ) -> None:
        """Create a new instance of the :py:class:`~Unilateral` class.

        The ``graph`` that represents the lymphatic system is given as a dictionary. Its
        keys are tuples of the form ``("tumor", "<tumor_name>")`` or
        ``("lnl", "<lnl_name>")``. The values are lists of strings that represent the
        names of the nodes that are connected to the node given by the key.

        Note:
            Do make sure the values in the dictionary are lists and not sets. Sets
            do not preserve the order of the elements and thus the order of the edges
            in the graph. This may lead to inconsistencies in the model.

        For example, the following graph represents a lymphatic system with one tumors
        and three lymph node levels:

        .. code-block:: python

            graph = {
                ("tumor", "T"): ["II", "III", "IV"],
                ("lnl", "II"): ["III"],
                ("lnl", "III"): ["IV"],
                ("lnl", "IV"): [],
            }

        The ``tumor_state`` is the initial (and unchangeable) state of the tumor. The
        states the LNLs can take on are given by ``allowed_states``. The default is
        a binary representation with ``allowed_states=[0, 1]``. For this, one can also
        use the classmethod :py:meth:`~Unilateral.binary`. For a trinary representation
        with ``allowed_states=[0, 1, 2]`` use the classmethod
        :py:meth:`~Unilateral.trinary`.

        The ``max_time`` parameter defines the latest possible time step for a
        diagnosis. In the HMM case, the probability disitrubtion over all hidden states
        is evolved from :math:`t=0` to ``max_time``. In the BN case, this parameter has
        no effect.
        """
        self.graph = graph.Representation(
            graph_dict=graph_dict,
            tumor_state=tumor_state,
            allowed_states=allowed_states,
            on_edge_change=[self.delete_transition_matrix],
        )

        if 0 >= max_time:
            raise ValueError("Latest diagnosis time `max_time` must be positive int")

        self.max_time = max_time


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
        return f"Unilateral with {len(self.graph._tumors)} tumors and {len(self.graph._lnls)} LNLs"


    def print_graph(self):
        """generates the a a visual chart of the spread model based on mermaid graph

        Returns:
            list: list with the string to create the mermaid graph and an url that directly leads to the graph
        """
        mermaid_graph = ('flowchart TD\n')
        for index, node in enumerate(self.graph.nodes):
            for edge in self.graph.nodes[index].out:
                line = f"{node.name} -->|{edge.spread_prob}| {edge.child.name} \n"
                mermaid_graph += line
        graphbytes = mermaid_graph.encode("ascii")
        base64_bytes = base64.b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")
        url="https://mermaid.ink/img/" + base64_string
        return mermaid_graph, url


    def print_info(self):
        """Print detailed information about the instance."""
        num_tumors = len(self.graph._tumors)
        num_lnls   = len(self.graph._lnls)
        string = (
            f"Unilateral lymphatic system with {num_tumors} tumor(s) "
            f"and {num_lnls} LNL(s).\n"
            + " ".join([f"{e} {e.spread_prob}%" for e in self.graph._tumor_edges]) + "\n" + " ".join([f"{e} {e.spread_prob}%" for e in self.graph._lnl_edges])
            + f"\n the growth probability is: {self.graph._growth_edges[0].spread_prob}" + f" the micro mod is {self.graph._lnl_edges[0].micro_mod}"
        )
        print(string)


    diag_time_dists = diagnose_times.Distributions()
    """Mapping of T-categories to the corresponding distributions over diagnose times.

    Every distribution is represented by a
    :py:class:`~diagnose_times.Distributions` object, which holds the
    parametrized and frozen versions of the probability mass function over the diagnose
    times. They are used to marginalize over the (generally unknown) diagnose times
    when computing e.g. the likelihood.
    """


    def get_params(self, as_dict: bool = False) -> dict[str, float] | list[float]:
        """Return a dictionary of all parameters and their currently set values.

        If ``as_dict`` is ``True``, the result is a dictionary with the names of the
        edge parameters as keys and their values as values. Otherwise, the result is a
        list of the values of the edge parameters in the order they appear in the
        graph.
        """
        result = {}
        for name, param in self.graph.edge_params.items():
            result[name] = param.get_param()

        for name, dist in self.diag_time_dists.items():
            for param_name, value in dist.get_params().items():
                result[f"{name}_{param_name}"] = value

        return result if as_dict else list(result.values())


    def _assign_via_args(self, new_params_args):
        """Assign parameters to egdes and to distributions via positional arguments."""
        objects = itertools.chain(
            (param for param in self.graph.edge_params.values()),
            (dist for dist in self.diag_time_dists.values()),
        )
        new_params_args = iter(new_params_args)
        while True:
            try:
                obj = next(objects)
                if isinstance(obj, params.Param):
                    obj.set_param(next(new_params_args))
                elif isinstance(obj, diagnose_times.Distribution):
                    kwargs = obj.get_params()
                    obj.set_params(**{key: next(new_params_args) for key in kwargs})
            except StopIteration:
                break


    def _assign_via_kwargs(self, new_params_kwargs):
        """Assign parameters to egdes and to distributions via keyword arguments."""
        for key, value in new_params_kwargs.items():
            t_stage, param_name = key.split("_", 1)
            if t_stage in self.diag_time_dists:
                self.diag_time_dists[t_stage].set_params(**{param_name: value})

            elif key == "growth":
                for edge in self.graph._growth_edges:
                    edge.spread_prob = value

            elif key == "micro_mod":
                for edge in self.graph._lnl_edges:
                    edge.micro_mod = value

            else:
                self.graph.edge_params[key].set_param(value)


    def assign_params(self, *new_params_args, **new_params_kwargs):
        """Assign new parameters to the model.

        The parameters can either be provided with positional arguments or as
        keyword arguments. The positional arguments must be in the following order:

        1. All spread probs from tumor to the LNLs
        2. The spread probs from LNL to LNL. If the model is trinary, the microscopic
           parameter is set right after the corresponding LNL's spread prob.
        3. The growth parameters for each trinary LNL. For a binary model,
           this is skipped.
        4. The parameters for the marginalizing distributions over diagnose times. Note
           that a distribution may take more than one parameter. So, if there are e.g.
           two T-stages with distributions over diagnose times, this step requires four
           arguments.

        The order of the keyword arguments obviously does not matter. If one wants to
        set the microscopic or growth parameters globally for all LNLs, the keyword
        arguments ``micro_mod`` and ``growth`` can be used for that.

        Note:
            Providing positional arguments does not allow using the global
            parameters ``micro_mod`` and ``growth``.

        Since the distributions over diagnose times may take more than one parameter,
        they can be provided as keyword arguments by appending their name to the
        corresponding T-stage, separated by an underscore. For example, a parameter
        ``foo`` for the T-stage ``early`` is set via the keyword argument ``early_foo``.

        Note:
            When using keyword arguments to set the parameters of the distributions
            over diagnose times, it is not possible to just use the name of the
            T-stage, even when the distribution only takes one parameter.

        Note:
            The keyword arguments override the positional arguments.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self._assign_via_args(new_params_args)
            self._assign_via_kwargs(new_params_kwargs)


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
        for i, lnl in enumerate(self.graph._lnls):
            trans_prob *= lnl.comp_trans_prob(new_state = newstate[i])
            if trans_prob == 0:
                break

        if assign:
            self.graph.set_state(newstate)

        return trans_prob


    modalities = modalities.ConfusionMatrices()
    """Dictionary storing diagnostic modalities and their specificity/sensitivity.

    The keys are the names of the modalities, e.g. "CT" or "pathology", the values are
    instances of the :py:class:`~lymph.descriptors.modalities.Modality` class. When
    setting the modality, the value can be a
    :py:class:`~lymph.descriptors.modalities.Modality` (or subclass) instance, a
    confusion matrix (``np.ndarray``) or a list/tuple with specificity and sensitivity.
    One can then access the confusion matrix of a modality.

    See Also:
        :py:class:`lymph.descriptors.modalities`
            The module managing this descriptor and the dictionary of modalities
            behind it.
    """


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
                for lnl in self.graph._lnls:
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


    def _gen_state_list(self):
        """Generates the list of (hidden) states."""
        allowed_states_list = []
        for lnl in self.graph._lnls:
            allowed_states_list.append(lnl.allowed_states)

        self._state_list = np.array(list(product(*allowed_states_list)))

    @property
    def state_list(self):
        """Return list of all possible hidden states.

        E.g., for three binary LNLs I, II, III, the first state would be where all LNLs
        are in state 0. The second state would be where LNL III is in state 1 and all
        others are in state 0, etc. The third represents the case where LNL II is in
        state 1 and all others are in state 0, etc. Essentially, it looks like binary
        counting:

        >>> model = Unilateral(graph={
        ...     ("tumor", "T"): ["I", "II" , "III"],
        ...     ("lnl", "I"): [],
        ...     ("lnl", "II"): ["I", "III"],
        ...     ("lnl", "III"): [],
        ... })
        >>> model.state_list
        array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 0],
               [0, 1, 1],
               [1, 0, 0],
               [1, 0, 1],
               [1, 1, 0],
               [1, 1, 1]])
        """
        try:
            return self._state_list
        except AttributeError:
            self._gen_state_list()
            return self._state_list


    def _gen_obs_list(self):
        """Generates the list of possible observations."""
        possible_obs_list = []
        for modality in self.modalities.values():
            possible_obs = np.arange(modality.confusion_matrix.shape[1])
            for _ in self.graph._lnls:
                possible_obs_list.append(possible_obs.copy())

        self._obs_list = np.array(list(product(*possible_obs_list)))

    @property
    def obs_list(self):
        """Return the list of all possible observations.

        They are ordered the same way as the :py:attr:`~Unilateral.state_list`, but
        additionally by modality. E.g., for two LNLs II, III and two modalities CT,
        pathology, the list would look like this:

        >>> model = Unilateral(graph={
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


    transition_matrix = matrix.Transition()
    """The matrix encoding the probabilities to transition from one state to another.

    This is the crucial object for modelling the evolution of the probabilistic
    system in the context of the hidden Markov model. It has the shape
    :math:`2^N \\times 2^N` where :math:`N` is the number of nodes in the graph.
    The :math:`i`-th row and :math:`j`-th column encodes the probability to transition
    from the :math:`i`-th state to the :math:`j`-th state. The states are ordered as
    in the `state_list`.

    This matrix is recomputed every time the parameters along the edges of the graph
    are changed.

    See Also:
        :py:class:`~lymph.descriptors.matrix.Transition`
            The class that implements the descriptor for the transition matrix.

    Example:

    >>> model = Unilateral(graph={
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
    def delete_transition_matrix(self):
        """Delete the transition matrix. Necessary to pass as callback."""
        del self.transition_matrix

    observation_matrix = matrix.Observation()
    """The matrix encoding the probabilities to observe a certain diagnosis.

    See Also:
        :py:class:`~lymph.descriptors.matrix.Observation`
            The class that implements the descriptor for the observation matrix.
    """
    def delete_observation_matrix(self):
        """Delete the observation matrix. Necessary to pass as callback."""
        del self.observation_matrix

    data_matrices = matrix.DataEncodings()
    """Dictionary with T-stages as keys and corresponding data matrices as values.

    A data matrix is a binary encding of which of the possible observational states
    agrees with the seen diagnosis of a patient. It accounts for missing involvement
    information on some LNLs and/or diagnostic modalities and thereby allows to
    marginalize over them.

    See Also:
        :py:class:`~lymph.descriptors.matrix.DataEncodings`
    """

    diagnose_matrices = matrix.Diagnoses()
    """Dictionary with T-stages as keys and corresponding diagnose matrices as values.

    Diagnose matrices are simply the dot product of the :py:attr:`~observation_matrix`
    and the :py:attr:`~data_matrices` for a given T-stage.

    See Also:
        :py:class:`~lymph.descriptors.matrix.Diagnoses`
    """


    @property
    def t_stages(self) -> set[str | int]:
        """Set of all valid T-stages in the model.

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

            for lnl in self.graph._lnls:
                if lnl.name not in patient_data[modality_name, side]:
                    raise ValueError(f"Involvement data for LNL {lnl} not found.")
                column = patient_data[modality_name, side, lnl.name]
                patient_data["_model", modality_name, lnl.name] = column

        patient_data["_model", "#", "t_stage"] = patient_data.apply(
            lambda row: mapping(row["tumor", "1", "t_stage"]), axis=1
        )

        for t_stage in self.diag_time_dists.keys():
            if t_stage not in patient_data["_model", "#", "t_stage"].values:
                warnings.warn(f"No data for T-stage {t_stage} found.")

        # Changes to the patient data require a recomputation of the data and
        # diagnose matrices. Deleting them will trigger this when they are next
        # accessed.
        del self.data_matrices
        del self.diagnose_matrices
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

        for t in range(1, self.max_time):
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
                self.graph.set_state(state)
                for node in self.graph._lnls:
                    state_dist[i] *= node.comp_bayes_net_prob()

            return state_dist


    def comp_obs_dist(self, t_stage: str) -> np.ndarray:
        """Compute the distribution over all possible observations for a given T-stage.

        Returns an array of probabilities for each possible complete observation. This
        entails multiplying the distribution over states as returned by the
        :py:meth:`~comp_state_dist` method with the :py:attr:`~observation_matrix`.

        Note that since the :py:attr:`~observation_matrix` can become very large, this
        method is not very efficient for inference. Instead, we compute the
        :py:attr:`~diagnose_matrices` from the :py:attr:`~observation_matrix` and
        the :py:attr:`~data_matrices` and use these to compute the likelihood.
        """
        state_dist = self.comp_state_dist(t_stage)
        return state_dist @ self.observation_matrix


    def _likelihood(
        self,
        mode: str = "HMM",
        log: bool = True,
    ) -> float:
        """Compute the (log-)likelihood of stored data, using the stored params."""
        if mode == "HMM":    # hidden Markov model
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

        elif mode == "BN":   # likelihood for the Bayesian network
            state_dist = self.comp_state_dist(mode=mode)
            patient_likelihoods = state_dist @ self.stacked_diagnose_matrix

            if log:
                llh = np.sum(np.log(patient_likelihoods))
            else:
                llh = np.prod(patient_likelihoods)

        return llh


    def likelihood(
        self,
        data: pd.DataFrame | None = None,
        given_params: list[float] | np.ndarray | dict[str, float] | None = None,
        log: bool = True,
        mode: str = "HMM"
    ) -> float:
        """Compute the (log-)likelihood of the ``data`` given the model (and params).

        If neither ``data`` nor the ``given_params`` are provided, it tries to compute
        the likelihood for the stored :py:attr:`~patient_data`,
        :py:attr:`~edge_params`, and the stored :py:attr:`~diag_time_dists`.

        Returns the log-likelihood if ``log`` is set to ``True``. The ``mode`` parameter
        determines whether the likelihood is computed for the hidden Markov model
        (``"HMM"``) or the Bayesian network (``"BN"``).
        """
        if data is not None:
            self.patient_data = data

        if given_params is None:
            return self._likelihood(mode, log)

        try:
            # all functions and methods called here should raise a ValueError if the
            # given parameters are invalid...
            if isinstance(given_params, dict):
                self.assign_params(**given_params)
            else:
                self.assign_params(*given_params)
        except ValueError:
            return -np.inf if log else 0.

        return self._likelihood(mode, log)


    def comp_posterior_state_dist(
        self,
        given_params: dict | None = None,
        given_diagnoses: PatternType | None = None,
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
        """
        if given_params is not None:
            self.assign_params(**given_params)

        if given_diagnoses is None:
            given_diagnoses = {}

        diagnose_encoding = np.array([True], dtype=bool)
        for modality in self.modalities.keys():
            diagnose_encoding = np.kron(
                diagnose_encoding,
                matrix.compute_encoding(
                    lnls=[lnl.name for lnl in self.graph._lnls],
                    pattern=given_diagnoses.get(modality, {}),
                ),
            )
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
        given_params: dict | None = None,
        given_diagnoses: dict[str, PatternType] | None = None,
        t_stage: str = "early",
        mode: str = "HMM",
        **_kwargs,
    ) -> float | np.ndarray:
        """Compute risk of a certain involvement, given a patient's diagnosis.

        If an ``involvement`` pattern of interest is provided, this method computes
        the risk of seeing just that pattern for the set of ``given_params`` and a
        dictionary of diagnoses for each modality.

        Using the ``mode`` parameter, the risk can be computed either for the hidden
        Markov model (``"HMM"``) or the Bayesian network (``"BN"``). In case of the
        Bayesian network mode, the ``t_stage`` parameter is ignored.

        See Also:
            :py:meth:`comp_posterior_state_dist`
        """
        posterior_state_dist = self.comp_posterior_state_dist(
            given_params, given_diagnoses, t_stage, mode,
        )

        if involvement is None:
            return posterior_state_dist

        # if a specific involvement of interest is provided, marginalize the
        # resulting vector of hidden states to match that involvement of
        # interest
        marginalize_over_states = matrix.compute_encoding(
            lnls=[lnl.name for lnl in self.graph._lnls],
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
        lnl_names = [lnl.name for lnl in self.graph._lnls]
        multi_cols = pd.MultiIndex.from_product([modality_names, lnl_names])

        # create DataFrame
        dataset = pd.DataFrame(drawn_obs, columns=multi_cols)
        dataset[('info', 't_stage')] = drawn_t_stages

        return dataset
