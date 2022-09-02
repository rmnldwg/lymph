import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from numpy.linalg import matrix_power as mat_pow

from .edge import Edge
from .node import Node
from .timemarg import MarginalizorDict


def change_base(
    number: int,
    base: int,
    reverse: bool = False,
    length: Optional[int] = None
) -> str:
    """Convert an integer into another base.

    Args:
        number: Number to convert
        base: Base of the resulting converted number
        reverse: If true, the converted number will be printed in reverse order.
        length: Length of the returned string. If longer than would be
            necessary, the output will be padded.

    Returns:
        The (padded) string of the converted number.
    """
    if number < 0:
        raise ValueError("Cannot convert negative numbers")
    if base > 16:
        raise ValueError("Base must be 16 or smaller!")
    elif base < 2:
        raise ValueError("There is no unary number system, base must be > 2")

    convertString = "0123456789ABCDEF"
    result = ''

    if number == 0:
        result += '0'
    else:
        while number >= base:
            result += convertString[number % base]
            number = number//base
        if number > 0:
            result += convertString[number]

    if length is None:
        length = len(result)
    elif length < len(result):
        length = len(result)
        warnings.warn("Length cannot be shorter than converted number.")

    pad = '0' * (length - len(result))

    if reverse:
        return result + pad
    else:
        return pad + result[::-1]


class Unilateral:
    """Class that models metastatic progression in a lymphatic system by
    representing it as a directed graph. The progression itself can be modelled
    via hidden Markov models (HMM) or Bayesian networks (BN).
    """
    def __init__(self, graph: Dict[Tuple[str], Set[str]], **_kwargs):
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
        | :math:`b_1` | :math:`b_2` | :math:`\\cdots` | :math:`b_n`  |
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


    @property
    def diag_time_dists(self) -> MarginalizorDict:
        """This property holds the probability mass functions for marginalizing over
        possible diagnose times for each T-stage.

        When setting this property, one may also provide a normal Python dict, in
        which case it tries to convert it to a :class:`MarginalizorDict`.

        See Also:
            :class:`MarginalzorDict`, :class:`Marginalizor`.
        """
        if not hasattr(self, "_diag_time_dists"):
            self._diag_time_dists = MarginalizorDict()
        return self._diag_time_dists

    @diag_time_dists.setter
    def diag_time_dists(self, new_dists: Union[dict, MarginalizorDict]):
        """Assign new :class:`MarginalizorDict` to this property. If it is a normal
        Python dictionary, tr to convert it into a :class:`MarginalizorDict`.
        """
        if isinstance(new_dists, MarginalizorDict):
            self._diag_time_dists = new_dists
        elif isinstance(new_dists, dict):
            warnings.warn("Trying to convert dictionary into MarginalizorDict.")
            guessed_max_t = len(new_dists.values()[0])
            self._diag_time_dists = MarginalizorDict(max_t=guessed_max_t)
            for t_stage, dist in new_dists.items():
                self._diag_time_dists[t_stage] = dist
        else:
            raise TypeError(
                f"Cannot use type {type(new_dists)} for marginalization over "
                "diagnose times."
            )


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
        diagnoses: Union[pd.Series, Dict[str, Dict[str, bool]]]
    ) -> float:
        """Compute the probability to observe a diagnose given the current
        state of the network.

        Args:
            diagnoses: Either a pandas ``Series`` object corresponding to one
                row of a patient data table, or a dictionary with keys of
                diagnostic modalities and values of dictionaries holding the
                observation for each LNL under the respective key.

        Returns:
            The probability of observing this particular combination of
            diagnoses, given the current state of the system.
        """
        prob = 1.
        for modality, spsn in self._spsn_tables.items():
            if modality in diagnoses:
                mod_diagnose = diagnoses[modality]
                for lnl in self.lnls:
                    try:
                        lnl_diagnose = mod_diagnose[lnl.name]
                    except KeyError:
                        continue
                    except IndexError as idx_err:
                        raise ValueError(
                            "diagnoses were not provided in the correct format"
                        ) from idx_err

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
        same order as the lymph node levels in the network/graph.
        """
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
                    observations[modality] = {
                        lnl.name: obs[n_lnl * k + i] for i,lnl in enumerate(self.lnls)
                    }
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
        except AttributeError as att_err:
            raise AttributeError(
                "No data has been loaded and hence no observation matrix has "
                "been computed."
            ) from att_err


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
        except AttributeError as att_err:
            raise AttributeError("No patient data has been loaded yet") from att_err

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

            modality_spsn: Dictionary of specificity :math:`s_P` and :math:`s_N`
                (in that order) for each observational/diagnostic modality. Can
                be ommitted if the modalities where already defined.

            mode: `"HMM"` for hidden Markov model and `"BN"` for Bayesian net.
        """
        if modality_spsn is not None:
            self.modalities = modality_spsn
        elif self.modalities == {}:
            raise ValueError("No diagnostic modalities have been defined yet!")

        # when first loading data with with different T-stages, and then loading a
        # dataset with fewer T-stages, the old diagnose matrices should not be preserved
        if hasattr(self, "_diagnose_matrices"):
            del self._diagnose_matrices

        # For the Hidden Markov Model
        if mode=="HMM":
            t_stages = list(set(data[("info", "t_stage")]))

            for stage in t_stages:
                table = data.loc[
                    data[('info', 't_stage')] == stage,
                    self._spsn_tables.keys()
                ]
                self._gen_diagnose_matrices(table, stage)
                if stage not in self.diag_time_dists:
                    warnings.warn(
                        "No distribution for marginalizing over diagnose times has "
                        f"been defined for T-stage {stage}. During inference, all "
                        "patients in this T-stage will be ignored."
                    )

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


    def check_and_assign(self, new_params: np.ndarray):
        """Check that the spread probability (rates) and the parameters for the
        marginalization over diagnose times are all within limits and assign them to
        the model.

        Args:
            new_params: The set of :attr:`spread_probs` and parameters to provide for
                updating the parametrized distributions over diagnose times.

        Warning:
            This method assumes that the parametrized distributions (instances of
            :class:`Marginalizor`) all raise a ``ValueError`` when provided with
            invalid parameters.
        """
        k = len(self.spread_probs)
        new_spread_probs = new_params[:k]
        new_marg_params = new_params[k:]

        try:
            self.diag_time_dists.update(new_marg_params)
        except ValueError as val_err:
            raise ValueError(
                "Parameters for marginalization over diagnose times are invalid"
            ) from val_err

        if new_spread_probs.shape != self.spread_probs.shape:
            raise ValueError(
                "Shape of provided spread parameters does not match network"
            )
        if np.any(0. > new_spread_probs) or np.any(new_spread_probs > 1.):
            raise ValueError(
                "Spread probs must be between 0 and 1"
            )

        self.spread_probs = new_spread_probs


    def _likelihood(
        self,
        mode: str = "HMM",
        log: bool = True,
    ) -> float:
        """
        Compute the (log-)likelihood of stored data, using the stored spread probs
        and parameters for the marginalizations over diagnose times (if the respective
        distributions are parametrized).

        This is the core method for computing the likelihood. The user-facing API calls
        it after doing some preliminary checks with the passed arguments.
        """
        # hidden Markov model
        if mode == "HMM":
            stored_t_stages = set(self.diagnose_matrices.keys())
            provided_t_stages = set(self.diag_time_dists.keys())
            t_stages = list(stored_t_stages.intersection(provided_t_stages))

            max_t = self.diag_time_dists.max_t
            evolved_model = self._evolve(t_last=max_t)

            llh = 0. if log else 1.
            for stage in t_stages:
                p = (
                    self.diag_time_dists[stage].pmf
                    @ evolved_model
                    @ self.diagnose_matrices[stage]
                )
                if log:
                    llh += np.sum(np.log(p))
                else:
                    llh *= np.prod(p)

        # likelihood for the Bayesian network
        elif mode == "BN":
            state_probs = np.ones(shape=(len(self.state_list),), dtype=float)

            for i, state in enumerate(self.state_list):
                self.state = state
                for node in self.lnls:
                    state_probs[i] *= node.bn_prob()

            p = state_probs @ self.diagnose_matrices["BN"]
            llh = np.sum(np.log(p)) if log else np.prod(p)

        return llh


    def likelihood(
        self,
        data: Optional[pd.DataFrame] = None,
        given_params: Optional[np.ndarray] = None,
        log: bool = True,
        mode: str = "HMM"
    ) -> float:
        """
        Compute (log-)likelihood of (already stored) data, given the probabilities of
        spread in the network and the parameters for the distributions used to
        marginalize over the diagnose times.

        Args:
            data: Table with rows of patients and columns of per-LNL involvment. See
                :meth:`load_data` for more details on how this should look like.

            given_params: The likelihood is a function of these parameters. They mainly
                consist of the :attr:`spread_probs` of the model. Any excess parameters
                will be used to update the parametrized distributions used for
                marginalizing over the diagnose times (see :attr:`diag_time_dists`).

            log: When ``True``, the log-likelihood is returned.

            mode: Compute the likelihood using the Bayesian network (``"BN"``) or
                the hidden Markv model (``"HMM"``). When using the Bayesian net, no
                marginalization over diagnose times is performed.

        Returns:
            The (log-)likelihood :math:`\\log{p(D \\mid \\theta)}` where :math:`D`
            is the data and :math:`\\theta` are the given parameters.
        """
        if data is not None:
            self.patient_data = data

        if given_params is None:
            return self._likelihood(mode, log)

        try:
            self.check_and_assign(given_params)
        except ValueError:
            return -np.inf if log else 0.

        return self._likelihood(mode, log)


    def risk(
        self,
        involvement: Optional[Union[dict, np.ndarray]] = None,
        given_params: Optional[np.ndarray] = None,
        given_diagnoses: Optional[Dict[str, dict]] = None,
        t_stage: str = "early",
        mode: str = "HMM",
        **_kwargs,
    ) -> Union[float, np.ndarray]:
        """Compute risk(s) of involvement given a specific (but potentially
        incomplete) diagnosis.

        Args:
            involvement: Specific hidden involvement one is interested in. If only parts
                of the state are of interest, the remainder can be masked with
                values ``None``. If specified, the functions returns a single
                risk.

            given_params: The risk is a function of these parameters. They mainly
                consist of the :attr:`spread_probs` of the model. Any excess parameters
                will be used to update the parametrized distributions used for
                marginalizing over the diagnose times (see :attr:`diag_time_dists`).

            given_diagnoses: Dictionary that can hold a potentially incomplete (mask
                with ``None``) diagnose for every available modality. Leaving
                out available modalities will assume a completely missing
                diagnosis.

            t_stage: The T-stage for which the risk should be computed. The attribute
                :attr:`diag_time_dists` must have a distribution for marginalizing
                over diagnose times stored for this T-stage.

            mode: Set to ``"HMM"`` for the hidden Markov model risk (requires
                the ``time_dist``) or to ``"BN"`` for the Bayesian network
                version.

        Returns:
            A single probability value if ``involvement`` is specified and an array
            with probabilities for all possible hidden states otherwise.
        """
        if given_params is not None:
            self.check_and_assign(given_params)

        if given_diagnoses is None:
            given_diagnoses = {}

        # vector containing P(Z=z|X)
        diagnose_probs = np.zeros(shape=len(self.state_list))
        for i,state in enumerate(self.state_list):
            self.state = state
            diagnose_probs[i] = self.comp_diagnose_prob(given_diagnoses)

        # vector P(X=x) of probabilities of arriving in state x, marginalized over time
        # HMM version
        if mode == "HMM":
            max_t = self.diag_time_dists.max_t
            state_probs = self._evolve(t_last=max_t)
            marg_state_probs = self.diag_time_dists[t_stage].pmf @ state_probs

        # BN version
        elif mode == "BN":
            marg_state_probs = np.ones(shape=(len(self.state_list)), dtype=float)
            for i, state in enumerate(self.state_list):
                self.state = state
                for node in self.lnls:
                    marg_state_probs[i] *= node.bn_prob()

        # multiply P(Z=z|X) * P(X) elementwise to get vector of joint probs P(Z=z,X)
        joint_diag_state = marg_state_probs * diagnose_probs

        # get marginal over X from joint
        marg_diagnose_prob = np.sum(joint_diag_state)

        # compute vector of probabilities for all possible involvements given
        # the specified diagnosis P(X|Z=z)
        post_state_probs =  joint_diag_state / marg_diagnose_prob

        if involvement is None:
            return post_state_probs

        # if a specific involvement of interest is provided, marginalize the
        # resulting vector of hidden states to match that involvement of
        # interest
        if isinstance(involvement, dict):
            involvement = np.array([involvement.get(lnl.name, None) for lnl in self.lnls])
        else:
            involvement = np.array(involvement)

        marg_states = np.zeros(shape=post_state_probs.shape, dtype=bool)
        for i,state in enumerate(self.state_list):
            marg_states[i] = np.all(np.equal(
                involvement, state,
                where=(involvement!=None),
                out=np.ones_like(state, dtype=bool)
            ))
        return marg_states @ post_state_probs


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
        stage_dist: Dict[str, float],
        **_kwargs,
    ) -> pd.DataFrame:
        """Generate/sample a pandas :class:`DataFrame` from the defined network
        using the samples and diagnostic modalities that have been set.

        Args:
            num_patients: Number of patients to generate.
            stage_dist: Probability to find a patient in a certain T-stage.
        """
        drawn_t_stages, drawn_diag_times = self.diag_time_dists.draw(
            dist=stage_dist, size=num_patients
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