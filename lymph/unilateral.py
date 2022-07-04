import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from numpy.linalg import matrix_power as mat_pow

from .edge import Edge
from .node import Node
from .utils import (
    HDFMixin,
    change_base,
    draw_diagnose_times,
    fast_binomial_pmf,
)


class Unilateral(HDFMixin):
    """Class that models metastatic progression in a lymphatic system by
    representing it as a directed graph. The progression itself can be modelled
    via hidden Markov models (HMM) or Bayesian networks (BN).
    """
    def __init__(self, graph: Dict[Tuple[str], Set[str]] = {}, **kwargs):
        """Initialize the underlying graph:

        Args:
            graph: Every key in this dictionary is a 2-tuple containing the
                type of the :class:`Node` ("tumor" or "lnl") and its name
                (arbitrary string). The corresponding value is a list of names
                this node should be connected to via an :class:`Edge`.
        """
        name_list = [tpl[1] for tpl in graph.keys()]
        name_set = {tpl[1] for tpl in graph.keys()}
        if len(name_list) != len(name_set):
            raise ValueError("No tumor and LNL can have the same name")

        self.nodes = []        # list of all nodes in the graph
        self.tumors = []       # list of nodes with type tumour
        self.lnls = []         # list of all lymph node levels

        for key in graph:
            self.nodes.append(Node(name=key[1], typ=key[0]))

        for node in self.nodes:
            if node.typ == "tumor":
                self.tumors.append(node)
            else:
                self.lnls.append(node)

        self.edges = []        # list of all edges connecting nodes in the graph
        self.base_edges = []   # list of edges, going out from tumors
        self.trans_edges = []  # list of edges, connecting LNLs

        for key, values in graph.items():
            for value in values:
                self.edges.append(Edge(self.find_node(key[1]),
                                       self.find_node(value)))

        for edge in self.edges:
            if edge.start.typ == "tumor":
                self.base_edges.append(edge)
            else:
                self.trans_edges.append(edge)


    def __str__(self):
        """Print info about the structure and parameters of the graph.
        """
        num_tumors = len(self.tumors)
        num_lnls   = len(self.lnls)
        string = (
            f"Unilateral lymphatic system with {num_tumors} tumor(s) "
            f"and {num_lnls} LNL(s).\n"
            + " ".join([f"{e}" for e in self.edges])
        )
        return string


    def find_node(self, name: str) -> Union[Node, None]:
        """Finds and returns a node with name ``name``.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        return None


    def find_edge(self, startname: str, endname: str) -> Union[Edge, None]:
        """Finds and returns the edge instance which has a parent node named
        ``startname`` and ends with node ``endname``.
        """
        for node in self.nodes:
            if node.name == startname:
                for o in node.out:
                    if o.end.name == endname:
                        return o
        return None


    @property
    def graph(self) -> Dict[Tuple[str, str], Set[str]]:
        """Lists the graph as it was provided when the system was created.
        """
        res = {}
        for node in self.nodes:
            res[(node.typ, node.name)] = [o.end.name for o in node.out]
        return res


    @property
    def state(self):
        """Return the currently set state of the system.
        """
        return np.array([lnl.state for lnl in self.lnls], dtype=int)

    @state.setter
    def state(self, newstate: np.ndarray):
        """Sets the state of the system to ``newstate``.
        """
        if len(newstate) < len(self.lnls):
            raise ValueError("length of newstate must match # of LNLs")

        for i, node in enumerate(self.lnls):  # only set lnl's states
            node.state = newstate[i]


    @property
    def base_probs(self):
        """The spread probablities parametrizing the edges that represent the
        lymphatic drainage from the tumor(s) to the individual lymph node
        levels. This array is composed of these elements:

        +-------------+-------------+-----------------+-------------+
        | :math:`b_1` | :math:`b_2` | :math:`\\cdots` | :math:`b_n` |
        +-------------+-------------+-----------------+-------------+

        Where :math:`n` is the number of edges between the tumor and the LNLs.

        Setting these requires an array with a length equal to the number of
        edges in the graph that start with a tumor node. After setting these
        values, the transition matrix - if it was precomputed - is deleted
        so it can be recomputed with the new parameters.
        """
        return np.array([edge.t for edge in self.base_edges], dtype=float)

    @base_probs.setter
    def base_probs(self, new_base_probs):
        """Set the spread probabilities for the connections from the tumor to
        the LNLs.
        """
        for i, edge in enumerate(self.base_edges):
            edge.t = new_base_probs[i]

        if hasattr(self, "_transition_matrix"):
            del self._transition_matrix


    @property
    def trans_probs(self):
        """Return the spread probablities of the connections between the lymph
        node levels. Here, "trans" stands for "transmission" (among the LNLs),
        not "transition" as in the transition to another state.

        Its shape is similar to the one of the :attr:`base_probs` but it lists
        the transmission probabilities :math:`t_{a \\rightarrow b}` in the
        order they were defined in the initial graph.

        When setting an array of length equal to the number of connections
        among the LNL is required. After setting the new values, the transition
        matrix - if previously computed - is deleted again, so that it will be
        recomputed with the new parameters.
        """
        return np.array([edge.t for edge in self.trans_edges], dtype=float)

    @trans_probs.setter
    def trans_probs(self, new_trans_probs):
        """Set the spread probabilities for the connections among the LNLs.
        """
        for i, edge in enumerate(self.trans_edges):
            edge.t = new_trans_probs[i]

        if hasattr(self, "_transition_matrix"):
            del self._transition_matrix


    @property
    def spread_probs(self) -> np.ndarray:
        """These are the probabilities of metastatic spread. They indicate how
        probable it is that a tumor or already cancerous lymph node level
        spreads further along a certain edge of the graph representing the
        lymphatic network.

        Setting these requires an array with a length equal to the number of
        edges in the network. It's essentially a concatenation of the
        :attr:`base_probs` and the :attr:`trans_probs`.
        """
        return np.concatenate([self.base_probs, self.trans_probs])

    @spread_probs.setter
    def spread_probs(self, new_spread_probs: np.ndarray):
        """Set the spread probabilities of the :class:`Edge` instances in the
        the network in the order they were created from the graph.
        """
        num_base_edges = len(self.base_edges)

        self.base_probs = new_spread_probs[:num_base_edges]
        self.trans_probs = new_spread_probs[num_base_edges:]


    def comp_transition_prob(
        self,
        newstate: List[int],
        acquire: bool = False
    ) -> float:
        """Computes the probability to transition to ``newstate``, given its
        current state.

        Args:
            newstate: List of new states for each LNL in the lymphatic
                system. The transition probability :math:`t` will be computed
                from the current states to these states.
            acquire: if ``True``, after computing and returning the probability,
                the system updates its own state to be ``newstate``.
                (default: ``False``)

        Returns:
            Transition probability :math:`t`.
        """
        res = 1.
        for i, lnl in enumerate(self.lnls):
            if not lnl.state:
                in_states = tuple(edge.start.state for edge in lnl.inc)
                in_weights = tuple(edge.t for edge in lnl.inc)
                res *= Node.trans_prob(in_states, in_weights)[newstate[i]]
            elif not newstate[i]:
                res = 0.
                break

        if acquire:
            self.state = newstate

        return res


    def comp_diagnose_prob(
        self,
        diagnoses: Union[pd.Series, Dict[str, list]]
    ) -> float:
        """Compute the probability to observe a diagnose given the current
        state of the network.

        Args:
            diagnoses: Either a pandas ``Series`` object corresponding to one
                row of a patient data table, or a dictionry with keys of
                diagnostic modalities and values of diagnoses for the respective
                modality.

        Returns:
            The probability of observing this particular combination of
            diagnoses, given the current state of the system.
        """
        prob = 1.
        is_pandas = type(diagnoses) == pd.Series

        for modality, spsn in self._spsn_tables.items():
            if modality in diagnoses:
                mod_diagnose = diagnoses[modality]
                if not is_pandas and len(mod_diagnose) != len(self.lnls):
                    raise ValueError(
                        "When providing a dictionary, the length of the "
                        "individual diagnoses must match the number of LNLs"
                    )
                for i,lnl in enumerate(self.lnls):
                    if is_pandas and lnl.name in mod_diagnose:
                        lnl_diagnose = mod_diagnose[lnl.name]
                    elif not is_pandas:
                        lnl_diagnose = mod_diagnose[i]
                    else:
                        continue

                    prob *= lnl.obs_prob(lnl_diagnose, spsn)
        return prob


    def _gen_state_list(self):
        """Generates the list of (hidden) states.
        """
        if not hasattr(self, "_state_list"):
            self._state_list = np.zeros(
                shape=(2**len(self.lnls), len(self.lnls)), dtype=int
            )
        for i in range(2**len(self.lnls)):
            self._state_list[i] = [
                int(digit) for digit in change_base(i, 2, length=len(self.lnls))
            ]

    @property
    def state_list(self):
        """Return list of all possible hidden states. They are arranged in the
        same order as the lymph node levels in the network/graph."""
        try:
            return self._state_list
        except AttributeError:
            self._gen_state_list()
            return self._state_list


    def _gen_obs_list(self):
        """Generates the list of possible observations.
        """
        n_obs = len(self._spsn_tables)

        if not hasattr(self, "_obs_list"):
            self._obs_list = np.zeros(
                shape=(2**(n_obs * len(self.lnls)), n_obs * len(self.lnls)),
                dtype=int
            )

        for i in range(2**(n_obs * len(self.lnls))):
            tmp = change_base(i, 2, reverse=False, length=n_obs * len(self.lnls))
            for j in range(len(self.lnls)):
                for k in range(n_obs):
                    self._obs_list[i,(j*n_obs)+k] = int(tmp[k*len(self.lnls)+j])

    @property
    def obs_list(self):
        """Return the list of all possible observations.
        """
        try:
            return self._obs_list
        except AttributeError:
            self._gen_obs_list()
            return self._obs_list


    def _gen_allowed_transitions(self):
        """Generate the allowed transitions.
        """
        self._allowed_transitions = {}
        for i in range(len(self.state_list)):
            self._allowed_transitions[i] = []
            for j in range(len(self.state_list)):
                if not np.any(np.greater(self.state_list[i,:],
                                         self.state_list[j,:])):
                    self._allowed_transitions[i].append(j)

    @property
    def allowed_transitions(self):
        """Return a dictionary that contains for each row :math:`i` of the
        transition matrix :math:`\\mathbf{A}` the column numbers :math:`j` for
        which the transtion probability :math:`P\\left( x_j \\mid x_i \\right)`
        is not zero due to the forbidden self-healing.

        For example: The hidden state ``[True, False]`` in a network with only
        one tumor and two LNLs (one involved, one healthy) corresponds to the
        index ``1`` and can only evolve into the state ``[True, True]``, which
        has index 3. So, the key-value pair for that particular hidden state
        would be ``1: [3]``.
        """
        try:
            return self._allowed_transitions
        except AttributeError:
            self._gen_allowed_transitions()
            return self._allowed_transitions


    def _gen_transition_matrix(self):
        """Generate the transition matrix :math:`\\mathbf{A}`, which contains
        the :math:`P \\left( S_{t+1} \\mid S_t \\right)`. :math:`\\mathbf{A}`
        is a square matrix with size ``(# of states)``. The lower diagonal is
        zero.
        """
        if not hasattr(self, "_transition_matrix"):
            shape = (2**len(self.lnls), 2**len(self.lnls))
            self._transition_matrix = np.zeros(shape=shape)

        for i,state in enumerate(self.state_list):
            self.state = state
            for j in self.allowed_transitions[i]:
                transition_prob = self.comp_transition_prob(self.state_list[j])
                self._transition_matrix[i,j] = transition_prob

    @property
    def A(self) -> np.ndarray:
        """Return the transition matrix :math:`\\mathbf{A}`, which contains the
        probability to transition from any state :math:`S_t` to any other state
        :math:`S_{t+1}` one timestep later:
        :math:`P \\left( S_{t+1} \\mid S_t \\right)`. :math:`\\mathbf{A}` is a
        square matrix with size ``(# of states)``. The lower diagonal is zero,
        because those entries correspond to transitions that would require
        self-healing.

        Warning:
            This will be deprecated in favour of :attr:`transition_matrix`.
        """
        warnings.warn(
            "The unintuitive `A` will be dropped for the more semantic "
            "`transition_matrix`.",
            DeprecationWarning
        )
        try:
            return self._transition_matrix
        except AttributeError:
            self._gen_transition_matrix()
            return self._transition_matrix

    @property
    def transition_matrix(self) -> np.ndarray:
        """Return the transition matrix :math:`\\mathbf{A}`, which contains the
        probability to transition from any state :math:`S_t` to any other state
        :math:`S_{t+1}` one timestep later:
        :math:`P \\left( S_{t+1} \\mid S_t \\right)`. :math:`\\mathbf{A}` is a
        square matrix with size ``(# of states)``. The lower diagonal is zero,
        because those entries correspond to transitions that would require
        self-healing.
        """
        try:
            return self._transition_matrix
        except AttributeError:
            self._gen_transition_matrix()
            return self._transition_matrix


    @property
    def modalities(self):
        """Return specificity & sensitivity stored in this :class:`System` for
        every diagnostic modality that has been defined.
        """
        try:
            modality_spsn = {}
            for mod, table in self._spsn_tables.items():
                modality_spsn[mod] = [table[0,0], table[1,1]]
            return modality_spsn

        except AttributeError:
            msg = ("No modality defined yet with specificity & sensitivity.")
            warnings.warn(msg)
            return {}


    @modalities.setter
    def modalities(self, modality_spsn: Dict[Any, List[float]]):
        """Given specificity :math:`s_P` & sensitivity :math:`s_N` of different
        diagnostic modalities, create a 2x2 matrix for every disgnostic
        modality that stores

        .. math::
            \\begin{pmatrix}
            s_P & 1 - s_N \\\\
            1 - s_P & s_N
            \\end{pmatrix}
        """

        if hasattr(self, "_observation_matrix"):
            del self._observation_matrix
        if hasattr(self, "_obs_list"):
            del self._obs_list

        self._spsn_tables = {}
        for mod, spsn in modality_spsn.items():
            if not isinstance(mod, str):
                msg = ("Modality names must be strings.")
                raise TypeError(msg)

            has_len_2 = len(spsn) == 2
            is_above_lb = np.all(np.greater_equal(spsn, 0.5))
            is_below_ub = np.all(np.less_equal(spsn, 1.))
            if not has_len_2 or not is_above_lb or not is_below_ub:
                msg = ("For each modality provide a list of two decimals "
                       "between 0.5 and 1.0 as specificity & sensitivity "
                       "respectively.")
                raise ValueError(msg)

            sp, sn = spsn
            self._spsn_tables[mod] = np.array([[sp     , 1. - sn],
                                               [1. - sp, sn     ]])


    def _gen_observation_matrix(self):
        """Generates the observation matrix :math:`\\mathbf{B}`, which contains
        the probabilities :math:`P \\left(D \\mid S \\right)` of any possible
        unique observation :math:`D` given any possible true hidden state
        :math:`S`. :math:`\\mathbf{B}` has the shape ``(# of states, # of
        possible observations)``.
        """
        n_lnl = len(self.lnls)

        if not hasattr(self, "_observation_matrix"):
            shape = (len(self.state_list), len(self.obs_list))
            self._observation_matrix = np.zeros(shape=shape)

        for i,state in enumerate(self.state_list):
            self.state = state
            for j,obs in enumerate(self.obs_list):
                observations = {}
                for k,modality in enumerate(self._spsn_tables):
                    observations[modality] = obs[n_lnl * k : n_lnl * (k+1)]
                self._observation_matrix[i,j] = self.comp_diagnose_prob(observations)

    @property
    def observation_matrix(self) -> np.ndarray:
        """Return the observation matrix :math:`\\mathbf{B}`. It encodes the
        probability :math:`P\\left( D \\mid S \\right)` to see a certain
        diagnose :math:`D`, given a particular true (but hidden) state :math:`S`.
        It is meant to be multiplied from the right onto the transition matrix
        :math:`\\mathbf{A}`.

        See Also:
            :attr:`transition_matrix`: The mentioned transition matrix
            :math:`\\mathbf{A}`.
        """
        try:
            return self._observation_matrix
        except AttributeError:
            self._gen_observation_matrix()
            return self._observation_matrix


    def _gen_diagnose_matrices(self, table: pd.DataFrame, t_stage: str):
        """Generate the matrix containing the probabilities to see the provided
        diagnose, given any possible hidden state. The resulting matrix has
        size :math:`2^N \\times M` where :math:`N` is the number of nodes in
        the graph and :math:`M` the number of patients.

        Args:
            table: pandas ``DataFrame`` containing rows of patients. Must have
                ``MultiIndex`` columns with two levels: First, the modalities
                and second, the LNLs.
            t_stage: The T-stage all the patients in ``table`` belong to.
        """
        if not hasattr(self, "_diagnose_matrices"):
            self._diagnose_matrices = {}

        shape = (len(self.state_list), len(table))
        self._diagnose_matrices[t_stage] = np.ones(shape=shape)

        for i,state in enumerate(self.state_list):
            self.state = state

            for j, (_, patient) in enumerate(table.iterrows()):
                patient_obs_prob = self.comp_diagnose_prob(patient)
                self._diagnose_matrices[t_stage][i,j] = patient_obs_prob


    @property
    def diagnose_matrices(self):
        try:
            return self._diagnose_matrices
        except AttributeError:
            raise AttributeError(
                "No data has been loaded and hence no observation matrix has "
                "been computed."
            )


    @property
    def patient_data(self):
        """Table with rows of patients. Must have a two-level
        :class:`MultiIndex` where the top-level has categories 'info' and the
        name of the available diagnostic modalities. Under 'info', the second
        level is only 't_stage', while under the modality, the names of the
        diagnosed lymph node levels are given as the columns. Such a table
        could look like this:

        +---------+----------------------+-----------------------+
        |  info   |         MRI          |          PET          |
        +---------+----------+-----------+-----------+-----------+
        | t_stage |    II    |    III    |    II     |    III    |
        +=========+==========+===========+===========+===========+
        | early   | ``True`` | ``False`` | ``True``  | ``False`` |
        +---------+----------+-----------+-----------+-----------+
        | late    | ``None`` | ``None``  | ``False`` | ``False`` |
        +---------+----------+-----------+-----------+-----------+
        | early   | ``True`` | ``True``  | ``True``  | ``None``  |
        +---------+----------+-----------+-----------+-----------+
        """
        try:
            return self._patient_data
        except AttributeError:
            raise AttributeError(
                "No patient data has been loaded yet"
            )

    @patient_data.setter
    def patient_data(self, patient_data: pd.DataFrame):
        """Load the patient data. For now, this just calls the :meth:`load_data`
        method, but at a later point, I would like to write a function here
        that generates the pandas :class:`DataFrame` from the internal matrix
        representation of the data.
        """
        self._patient_data = patient_data.copy()
        self.load_data(patient_data)


    def load_data(
        self,
        data: pd.DataFrame,
        t_stages: Optional[List[int]] = None,
        modality_spsn: Optional[Dict[str, List[float]]] = None,
        mode: str = "HMM",
    ):
        """
        Transform tabular patient data (:class:`pd.DataFrame`) into internal
        representation, consisting of one or several matrices
        :math:`\\mathbf{C}_{T}` that can marginalize over possible diagnoses.

        Args:
            data: Table with rows of patients. Must have a two-level
                :class:`MultiIndex` where the top-level has categories 'info'
                and the name of the available diagnostic modalities. Under
                'info', the second level is only 't_stage', while under the
                modality, the names of the diagnosed lymph node levels are
                given as the columns. Such a table could look like this:

                +---------+----------------------+-----------------------+
                |  info   |         MRI          |          PET          |
                +---------+----------+-----------+-----------+-----------+
                | t_stage |    II    |    III    |    II     |    III    |
                +=========+==========+===========+===========+===========+
                | early   | ``True`` | ``False`` | ``True``  | ``False`` |
                +---------+----------+-----------+-----------+-----------+
                | late    | ``None`` | ``None``  | ``False`` | ``False`` |
                +---------+----------+-----------+-----------+-----------+
                | early   | ``True`` | ``True``  | ``True``  | ``None``  |
                +---------+----------+-----------+-----------+-----------+

            t_stages: List of T-stages that should be included in the learning
                process. If ommitted, the list of T-stages is extracted from
                the :class:`DataFrame`

            modality_spsn: Dictionary of specificity :math:`s_P` and :math:`s_N`
                (in that order) for each observational/diagnostic modality. Can
                be ommitted if the modalities where already defined.

            mode: `"HMM"` for hidden Markov model and `"BN"` for Bayesian net.
        """
        if modality_spsn is not None:
            self.modalities = modality_spsn
        elif self.modalities == {}:
            msg = ("No diagnostic modalities have been defined yet!")
            raise ValueError(msg)

        # For the Hidden Markov Model
        if mode=="HMM":
            if t_stages is None:
                t_stages = list(set(data[("info", "t_stage")]))

            for stage in t_stages:
                table = data.loc[
                    data[('info', 't_stage')] == stage,
                    self._spsn_tables.keys()
                ]
                self._gen_diagnose_matrices(table, stage)

        # For the Bayesian Network
        elif mode=="BN":
            table = data[self._spsn_tables.keys()]
            stage = "BN"
            self._gen_diagnose_matrices(table, stage)


    def _evolve(
        self, t_first: int = 0, t_last: Optional[int] = None
    ) -> np.ndarray:
        """Evolve hidden Markov model based system over time steps. Compute
        :math:`p(S \\mid t)` where :math:`S` is a distinct state and :math:`t`
        is the time.

        Args:
            t_first: First time-step that should be in the list of returned
                involvement probabilities.

            t_last: Last time step to consider. This function computes
                involvement probabilities for all :math:`t` in between
                ``t_frist`` and ``t_last``. If ``t_first == t_last``,
                :math:`p(S \\mid t)` is computed only at that time.

        Returns:
            A matrix with the values :math:`p(S \\mid t)` for each time-step.

        :meta public:
        """
        # All healthy state at beginning
        start_state = np.zeros(shape=len(self.state_list), dtype=float)
        start_state[0] = 1.

        # compute involvement at first time-step
        state = start_state @ mat_pow(self.transition_matrix, t_first)

        if t_last is None:
            return state

        len_time_range = t_last - t_first
        if len_time_range < 0:
            msg = ("Starting time must be smaller than ending time.")
            raise ValueError(msg)

        state_probs = np.zeros(
            shape=(len_time_range + 1, len(self.state_list)),
            dtype=float
        )

        # compute subsequent time-steps, effectively incrementing time until end
        for i in range(len_time_range):
            state_probs[i] = state
            state = state @ self.transition_matrix

        state_probs[-1] = state

        return state_probs


    def _are_valid_(self, new_spread_probs: np.ndarray) -> bool:
        """Check that the spread probability (rates) are all within limits.
        """
        if new_spread_probs.shape != self.spread_probs.shape:
            msg = ("Shape of provided spread parameters does not match network")
            raise ValueError(msg)
        if np.any(0. > new_spread_probs):
            return False
        if np.any(new_spread_probs > 1.):
            return False

        return True


    def _log_likelihood(
        self,
        t_stages: Optional[List[Any]] = None,
        diag_times: Optional[Dict[Any, int]] = None,
        max_t: Optional[int] = 10,
        time_dists: Optional[Dict[Any, np.ndarray]] = None,
        mode: str = "HMM",
    ) -> float:
        """
        Compute the log-likelihood of data, using the stored spread probs.
        This method mainly exists so that the checking and assigning of the
        spread probs can be skipped.
        """
        # hidden Markov model
        if mode == "HMM":
            if t_stages is None:
                t_stages = list(self.diagnose_matrices.keys())

            state_probs = {}

            if diag_times is not None:
                if len(diag_times) != len(t_stages):
                    raise ValueError(
                        "One diagnose time must be provided for each T-stage."
                    )

                for stage in t_stages:
                    diag_time = np.around(diag_times[stage]).astype(int)
                    if diag_time > max_t:
                        return -np.inf
                    state_probs[stage] = self._evolve(diag_time)

            elif time_dists is not None:
                if len(time_dists) != len(t_stages):
                    raise ValueError(
                        "One distribution over diagnose times must be provided "
                        "for each T-stage."
                    )

                # subtract 1, to also consider healthy starting state (t = 0)
                max_t = len(time_dists[t_stages[0]]) - 1

                for stage in t_stages:
                    state_probs[stage] = time_dists[stage] @ self._evolve(t_last=max_t)

            else:
                raise ValueError(
                    "Either provide a list of diagnose times for each T-stage "
                    "or a distribution over diagnose times for each T-stage."
                )

            llh = 0.
            for stage in t_stages:
                p = state_probs[stage] @ self.diagnose_matrices[stage]
                llh += np.sum(np.log(p))

        # likelihood for the Bayesian network
        elif mode == "BN":
            state_probs = np.ones(shape=(len(self.state_list),), dtype=float)

            for i, state in enumerate(self.state_list):
                self.state = state
                for node in self.lnls:
                    state_probs[i] *= node.bn_prob()

            p = state_probs @ self.diagnose_matrices["BN"]
            llh = np.sum(np.log(p))

        return llh

    def log_likelihood(
        self,
        spread_probs: np.ndarray,
        t_stages: Optional[List[Any]] = None,
        diag_times: Optional[Dict[Any, int]] = None,
        max_t: Optional[int] = 10,
        time_dists: Optional[Dict[Any, np.ndarray]] = None,
        mode: str = "HMM"
    ) -> float:
        """
        Compute log-likelihood of (already stored) data, given the spread
        probabilities and either a discrete diagnose time or a distribution to
        use for marginalization over diagnose times.

        Args:
            spread_probs: Spread probabiltites from the tumor to the LNLs, as
                well as from (already involved) LNLs to downsream LNLs.

            t_stages: List of T-stages that are also used in the data to denote
                how advanced the primary tumor of the patient is. This does not
                need to correspond to the clinical T-stages 'T1', 'T2' and so
                on, but can also be more abstract like 'early', 'late' etc. If
                not given, this will be inferred from the loaded data.

            diag_times: For each T-stage, one can specify with what time step
                the likelihood should be computed. If this is set to `None`,
                and a distribution over diagnose times `time_dists` is provided,
                the function marginalizes over diagnose times.

            max_t: Latest possible diagnose time. This is only used to return
                `-np.inf` in case one of the `diag_times` exceeds this value.

            time_dists: Distribution over diagnose times that can be used to
                compute the likelihood of the data, given the spread
                probabilities, but marginalized over the time of diagnosis. If
                set to `None`, a diagnose time must be explicitly set for each
                T-stage.

            mode: Compute the likelihood using the Bayesian network (`"BN"`) or
                the hidden Markv model (`"HMM"`). When using the Bayesian net,
                the inputs `t_stages`, `diag_times`, `max_t` and `time_dists`
                are ignored.

        Returns:
            The log-likelihood :math:`\\log{p(D \\mid \\theta)}` where :math:`D`
            is the data and :math:`\\theta` is the tuple of spread probabilities
            and diagnose times or distributions over diagnose times.
        """
        if not self._are_valid_(spread_probs):
            return -np.inf

        self.spread_probs = spread_probs

        return self._log_likelihood(
            t_stages=t_stages,
            diag_times=diag_times,
            max_t=max_t,
            time_dists=time_dists,
            mode=mode,
        )


    def marginal_log_likelihood(
        self,
        theta: np.ndarray,
        t_stages: Optional[List[Any]] = None,
        time_dists: Dict[Any, np.ndarray] = {}
    ) -> float:
        """
        Compute the likelihood of the (already stored) data, given the spread
        parameters, marginalized over time of diagnosis via time distributions.

        Args:
            theta: Set of parameters, consisting of the base probabilities
                :math:`b` and the transition probabilities :math:`t`.

            t_stages: List of T-stages that should be included in the learning
                process.

            time_dists: Distribution over the probability of diagnosis at
                different times :math:`t` given T-stage.

        Returns:
            The log-likelihood of the data, given te spread parameters.
        """
        return self.log_likelihood(
            theta, t_stages,
            diag_times=None, time_dists=time_dists,
            mode="HMM"
        )


    def binom_marg_log_likelihood(
        self,
        theta: np.ndarray,
        t_stages: List[Any],
        max_t: int = 10
    ) -> float:
        """
        Compute the likelihood of the (already stored) data, given the spread
        parameters, marginalized over time of diagnosis via binomial
        distributions.

        Args:
            theta: Set of parameters, consisting of the base probabilities
                :math:`b` and the transition probabilities :math:`t`.

            t_stages: List of T-stages that should be included in the learning.

            max_t: Latest possible diagnose time.

        Returns:
            The log-likelihood of the data, given te spread parameters and binomial
            time distributions.
        """
        # splitting theta into spread parameters and...
        len_spread_probs = len(theta) - len(t_stages)
        spread_probs = theta[:len_spread_probs]
        # ...p-values for the binomial distribution
        p = theta[len_spread_probs:]

        if np.any(np.greater(p, 1.)) or np.any(np.less(p, 0.)):
            return -np.inf

        t = np.arange(max_t + 1)
        time_dists = {}
        for i,stage in enumerate(t_stages):
            time_dists[stage] = fast_binomial_pmf(t, max_t, p[i])

        return self.marginal_log_likelihood(
            spread_probs, t_stages,
            time_dists=time_dists
        )


    def time_log_likelihood(
        self,
        theta: np.ndarray,
        t_stages: List[Any],
        max_t: int = 10
    ) -> float:
        """
        Compute likelihood given the spread parameters and the time of diagnosis
        for each T-stage.

        Args:
            theta: Set of parameters, consisting of the spread probabilities
                (as many as the system has :class:`Edge` instances) and the
                time of diagnosis for all T-stages.

            t_stages: keywords of T-stages that are present in the dictionary of
                C matrices and the previously loaded dataset.

            max_t: Largest accepted time-point.

        Returns:
            The likelihood of the data, given the spread parameters as well as
            the diagnose time for each T-stage.
        """
        # splitting theta into spread parameters and...
        len_spread_probs = len(theta) - len(t_stages)
        spread_probs = theta[:len_spread_probs]
        # ...diagnose times for each T-stage
        tmp = theta[len_spread_probs:]
        diag_times = {t_stages[t]: tmp[t] for t in range(len(t_stages))}

        return self.log_likelihood(
            spread_probs, t_stages,
            diag_times=diag_times, max_t=max_t, time_dists=None,
            mode="HMM"
        )


    def risk(
        self,
        spread_probs: Optional[np.ndarray] = None,
        inv: Optional[np.ndarray] = None,
        diagnoses: Dict[str, np.ndarray] = {},
        diag_time: Optional[int] = None,
        time_dist: Optional[np.ndarray] = None,
        mode: str = "HMM"
    ) -> Union[float, np.ndarray]:
        """Compute risk(s) of involvement given a specific (but potentially
        incomplete) diagnosis.

        Args:
            spread_probs: Set of new spread parameters. If not given (``None``),
                the currently set parameters will be used.

            inv: Specific hidden involvement one is interested in. If only parts
                of the state are of interest, the remainder can be masked with
                values ``None``. If specified, the functions returns a single
                risk.

            diagnoses: Dictionary that can hold a potentially incomplete (mask
                with ``None``) diagnose for every available modality. Leaving
                out available modalities will assume a completely missing
                diagnosis.

            diag_time: Time of diagnosis. Either this or the ``time_dist`` to
                marginalize over diagnose times must be given.

            time_dist: Distribution to marginalize over diagnose times. Either
                this, or the ``diag_time`` must be given.

            mode: Set to ``"HMM"`` for the hidden Markov model risk (requires
                the ``time_dist``) or to ``"BN"`` for the Bayesian network
                version.

        Returns:
            A single probability value if ``inv`` is specified and an array
            with probabilities for all possible hidden states otherwise.
        """
        # assign spread_probs to system or use the currently set one
        if spread_probs is not None:
            self.spread_probs = spread_probs

        # create one large diagnose vector from the individual modalitie's
        # diagnoses
        obs = np.array([])
        for mod in self._spsn_tables:
            if mod in diagnoses:
                obs = np.append(obs, diagnoses[mod])
            else:
                obs = np.append(obs, np.array([None] * len(self.lnls)))

        # vector of probabilities of arriving in state x, marginalized over time
        # HMM version
        if mode == "HMM":
            if diag_time is not None:
                pX = self._evolve(diag_time)

            elif time_dist is not None:
                max_t = len(time_dist)
                state_probs = self._evolve(t_last=max_t-1)
                pX = time_dist @ state_probs

            else:
                msg = ("Either diagnose time or distribution to marginalize "
                       "over it must be given.")
                raise ValueError(msg)

        # BN version
        elif mode == "BN":
            pX = np.ones(shape=(len(self.state_list)), dtype=float)
            for i, state in enumerate(self.state_list):
                self.state = state
                for node in self.lnls:
                    pX[i] *= node.bn_prob()

        # compute the probability of observing a diagnose z and being in a
        # state x which is P(z,x) = P(z|x)P(x). Do that for all combinations of
        # x and z and put it in a matrix
        pZX = self.observation_matrix.T * pX

        # vector of probabilities for seeing a diagnose z
        pZ = pX @ self.observation_matrix

        # build vector to marginalize over diagnoses
        cZ = np.zeros(shape=(len(pZ)), dtype=bool)
        for i,complete_obs in enumerate(self.obs_list):
            cZ[i] = np.all(np.equal(obs, complete_obs,
                                    where=(obs!=None),
                                    out=np.ones_like(obs, dtype=bool)))

        # compute vector of probabilities for all possible involvements given
        # the specified diagnosis
        res =  cZ @ pZX / (cZ @ pZ)

        if inv is None:
            return res
        else:
            # if a specific involvement of interest is provided, marginalize the
            # resulting vector of hidden states to match that involvement of
            # interest
            inv = np.array(inv)
            cX = np.zeros(shape=res.shape, dtype=bool)
            for i,state in enumerate(self.state_list):
                cX[i] = np.all(np.equal(inv, state,
                                        where=(inv!=None),
                                        out=np.ones_like(state, dtype=bool)))
            return cX @ res


    def _draw_patient_diagnoses(
        self,
        diag_times: List[int],
    ) -> np.ndarray:
        """Draw random possible observations for a list of T-stages and
        diagnose times.

        Args:
            diag_times: List of diagnose times for each patient who's diagnose
                is supposed to be drawn.
        """
        max_t = np.max(diag_times)

        # use the drawn diagnose times to compute probabilities over states and
        # diagnoses
        per_time_state_probs = self._evolve(t_last=max_t)
        per_patient_state_probs = per_time_state_probs[diag_times]
        per_patient_obs_probs = per_patient_state_probs @ self.observation_matrix

        # then, draw a diagnose from the possible ones
        obs_idx = np.arange(len(self.obs_list))
        drawn_obs_idx = [
            np.random.choice(obs_idx, p=obs_prob)
            for obs_prob in per_patient_obs_probs
        ]
        drawn_obs = self.obs_list[drawn_obs_idx].astype(bool)
        return drawn_obs


    def generate_dataset(
        self,
        num_patients: int,
        stage_dist: List[float],
        diag_times: Optional[Dict[Any, int]] = None,
        time_dists: Optional[Dict[Any, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Generate/sample a pandas :class:`DataFrame` from the defined network
        using the samples and diagnostic modalities that have been set.

        Args:
            num_patients: Number of patients to generate.
            stage_dist: Probability to find a patient in a certain T-stage.
            diag_times: For each T-stage, one can specify until which time step
                the corresponding patients should be evolved. If this is set to
                ``None``, and a distribution over diagnose times ``time_dists``
                is provided, the diagnose time is drawn from the ``time_dist``.
            time_dists: Distributions over diagnose times that can be used to
                draw a diagnose time for the respective T-stage. If ``None``,
                ``diag_times`` must be provided.
        """
        drawn_t_stages, drawn_diag_times = draw_diagnose_times(
            num_patients=num_patients,
            stage_dist=stage_dist,
            diag_times=diag_times,
            time_dists=time_dists
        )

        drawn_obs = self._draw_patient_diagnoses(drawn_diag_times)

        # construct MultiIndex for dataset from stored modalities
        modalities = list(self.modalities.keys())
        lnl_names = [lnl.name for lnl in self.lnls]
        multi_cols = pd.MultiIndex.from_product([modalities, lnl_names])

        # create DataFrame
        dataset = pd.DataFrame(drawn_obs, columns=multi_cols)
        dataset[('info', 't_stage')] = drawn_t_stages

        return dataset


class System(Unilateral):
    """Class kept for compatibility after renaming to :class:`Unilateral`.

    See Also:
        :class:`Unilateral`
    """
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "This class has been renamed to `Unilateral`.",
            DeprecationWarning
        )

        super().__init__(*args, **kwargs)