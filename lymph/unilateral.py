import numpy as np
from numpy.linalg import matrix_power as mat_pow
import scipy as sp 
import scipy.stats
import pandas as pd
import warnings
from typing import Union, Optional, List, Dict, Any

from .node import Node
from .edge import Edge



def toStr(n: int, 
          base: int, 
          rev: bool = False, 
          length: Optional[int] = None) -> str:
    """Function that converts an integer into another base.
    
    Args:
        n: Number to convert

        base: Base of the resulting converted number

        rev: If true, the converted number will be printed in reverse 
            order. (default: ``False``)

        length: Length of the returned string.
            (default: ``None``)

    Returns:
        The (padded) string of the converted number.
    """
    
    if base > 16:
        raise ValueError("Base must be 16 or smaller!")
        
    convertString = "0123456789ABCDEF"
    result = ''
    while n >= base:
        result = result + convertString[n % base]
        n = n//base
    if n > 0:
        result = result + convertString[n]
        
    if length is None:
        length = len(result)
    elif length < len(result):
        length = len(result)
        warnings.warn("Length cannot be shorter than converted number.")
        
    pad = '0' * (length - len(result))
        
    if rev:
        return result + pad
    else:
        return pad + result[::-1]



class System(object):
    """Class that models metastatic progression in a lymphatic system by 
    representing it as a directed graph. The progression itself can be modelled 
    via hidden Markov models (HMM) or Bayesian networks (BN). 

    Args:
        graph: Every key in this dictionary is a 2-tuple containing the type of 
            the :class:`Node` ("tumor" or "lnl") and its name (arbitrary 
            string). The corresponding value is a list of names this node should 
            be connected to via an :class:`Edge`
    """
    def __init__(self, 
                 graph: dict = {}):

        self.tumors = []    # list of nodes with type tumour
        self.lnls = []      # list of all lymph node levels
        self.nodes = []     # list of all nodes in the graph
        self.edges = []     # list of all edges connecting nodes in the graph
        
        for key in graph:
            self.nodes.append(Node(name=key[1], typ=key[0]))
            
        for node in self.nodes:
            if node.typ == "tumor":
                self.tumors.append(node)
            else:
                self.lnls.append(node)

        for key, values in graph.items():
            for value in values:
                self.edges.append(Edge(self.find_node(key[1]), 
                                       self.find_node(value)))

        self._gen_state_list()
        self._gen_mask()


    def __str__(self):
        """Print info about the structure and parameters of the graph.
        """
        string = "Tumor(s):\n"
        for tumor in self.tumors:
            if tumor.typ != "tumor":
                raise RuntimeError("Tumor node is not of type tumor")
            
            prefix = tumor.name + " ---"
            for o in tumor.out:
                string += f"\t{prefix} {o.t * 100: >4.1f} % --> {o.end.name}\n"
                prefix = "".join([" "] * (len(tumor.name) + 1)) + "`--"
                
        longest = 0
        for lnl in self.lnls:
            if len(lnl.name) > longest:
                longest = len(lnl.name)

        string += "\nLNL(s):\n"
        for lnl in self.lnls:
            if lnl.typ != "lnl":
                raise RuntimeError("LNL node is not of type LNL")

            prefix = lnl.name + " ---" + ("-" * (longest - len(lnl.name)))
            for o in lnl.out:
                string += f"\t{prefix} {o.t * 100: >4.1f} % --> {o.end.name}\n"
                prefix = " " * (len(lnl.name) + 1) + "`--" + ("-" * (longest - len(lnl.name)))
                
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


    def get_graph(self) -> dict:
        """Lists the graph as it was provided when the system was created.
        """
        res = []
        for node in self.nodes:
            out = []
            for o in node.out:
                out.append(o.end.name)
            res.append((node.name, out))
            
        return dict(res)


    def list_edges(self) -> List[Edge]:
        """Lists all edges of the system with its corresponding start and end 
        nodes.
        """
        res = []
        for edge in self.edges:
            res.append((edge.start.name, edge.end.name, edge.t))
            
        return res


    def set_state(self, newstate: np.ndarray):
        """Sets the state of the system to ``newstate``.
        """
        if len(newstate) != len(self.lnls):
            raise ValueError("length of newstate must match # of LNLs")
        
        for i, node in enumerate(self.lnls):  # only set lnl's states
            node.state = int(newstate[i])


    def get_theta(self) -> np.ndarray:
        """Return the spread probabilities of the :class:`Edge` instances in 
        the network in the order they appear in the graph.
        """
        theta = np.zeros(shape=(len(self.edges)))
        for i, edge in enumerate(self.edges):
            theta[i] = edge.t

        return theta


    def set_theta(self, 
                  theta: np.ndarray, 
                  mode: str = "HMM"):
        """Set the spread probabilities of the :class:`Edge` instances in the 
        the network int he order they were created from the graph.

        Args:
            theta: The new parameters that should be fed into the system. They 
                all represent the transition probabilities along the edges of 
                the network and will be set in the order they appear in the 
                graph dictionary. This includes the spread probabilities from 
                the primary tumour to the LNLs, as well as the spread among the 
                LNLs.

            mode: For mode "BN" the transition matrix :math:`\\mathbf{A}` is 
                not needed and its computation is skipped.
        """
        if len(theta) != len(self.edges):
            raise ValueError(f"# of parameters ({len(theta)}) must match # of "
                             f"edges ({len(self.edges)})")
        
        for i, edge in enumerate(self.edges):
            edge.t = theta[i]

        if mode=="HMM":
            try:
                self._gen_A()
            except AttributeError:
                self.A = np.zeros(shape=(len(self.state_list), 
                                         len(self.state_list)))
                self._gen_A()
            

    def trans_prob(self, 
                   newstate: List[int], 
                   log: bool = False, 
                   acquire: bool = False) -> float:
        """Computes the probability to transition to newstate, given its 
        current state.

        Args:
            newstate: List of new states for each LNL in the lymphatic 
                system. The transition probability :math:`t` will be computed 
                from the current states to these states.

            log: if ``True``, the log-probability is computed. 
                (default: ``False``)

            acquire: if ``True``, after computing and returning the probability, 
                the system updates its own state to be ``newstate``. 
                (default: ``False``)

        Returns:
            Transition probability :math:`t`.
        """
        if len(newstate) != len(self.lnls):
            raise ValueError("length of newstate must match # of LNLs")
        
        if not log:
            res = 1.
            for i, lnl in enumerate(self.lnls):
                res *= lnl.trans_prob(log=log)[newstate[i]]
        else:
            res = 0.
            for i, lnl in enumerate(self.lnls):
                res += lnl.trans_prob(log=log)[newstate[i]]

        if acquire:
            self.set_state(newstate)

        return res


    def obs_prob(self, 
                 diagnoses_dict: Dict[str, List[int]], 
                 log: bool = False) -> float:
        """Computes the probability to see certain diagnoses, given the 
        system's current state.

        Args:
            diagnoses_dict: Dictionary of diagnoses (one for each diagnostic 
                modality). A diagnose must be an array of integers that is as 
                long as the the system has LNLs.

            log: If ``True``, the log probability is computed. 
                (default: ``False``)

        Returns:
            The probability to see the given diagnoses.
        """  
        if not log:
            res = 1.
        else:
            res = 0.
            
        for modality, diagnoses in diagnoses_dict.items():
            if len(diagnoses) != len(self.lnls):
                raise ValueError("length of observations must match @ of LNLs")

            for i, lnl in enumerate(self.lnls):
                if not log:
                    res *= lnl.obs_prob(obs=diagnoses[i], 
                                        obstable=self._modality_dict[modality], 
                                        log=log)
                else:
                    res += lnl.obs_prob(obs=diagnoses[i],
                                        obstable=self._modality_dict[modality],
                                        log=log)
        return res


    def _gen_state_list(self):
        """Generates the list of (hidden) states.
        """                
        # every LNL can be either healthy (state=0) or involved (state=1). 
        # Hence, the number of different possible states is 2 to the power of 
        # the # of lymph node levels.
        self.state_list = np.zeros(shape=(2**len(self.lnls), len(self.lnls)), 
                                   dtype=int)
        
        for i in range(2**len(self.lnls)):
            tmp = toStr(i, 2, rev=False, length=len(self.lnls))
            for j in range(len(self.lnls)):
                self.state_list[i,j] = int(tmp[j])


    def _gen_obs_list(self):
        """Generates the list of possible observations.
        """
        n_obs = len(self._modality_dict)
        
        self.obs_list = np.zeros(shape=(2**(n_obs * len(self.lnls)), 
                                        n_obs * len(self.lnls)), dtype=int)
        for i in range(2**(n_obs * len(self.lnls))):
            tmp = toStr(i, 2, rev=False, length=n_obs * len(self.lnls))
            for j in range(len(self.lnls)):
                for k in range(n_obs):
                    self.obs_list[i,(j*n_obs)+k] = int(tmp[k*len(self.lnls)+j])


    def _gen_mask(self):
        """Generates a dictionary that contains for each row of 
        :math:`\\mathbf{A}` those indices where :math:`\\mathbf{A}` is NOT zero.
        """
        self._idx_dict = {}
        for i in range(len(self.state_list)):
            self._idx_dict[i] = []
            for j in range(len(self.state_list)):
                if not np.any(np.greater(self.state_list[i,:], 
                                         self.state_list[j,:])):
                    self._idx_dict[i].append(j)


    def _gen_A(self):
        """Generates the transition matrix :math:`\\mathbf{A}`, which contains 
        the :math:`P \\left( X_{t+1} \\mid X_t \\right)`. :math:`\\mathbf{A}` 
        is a square matrix with size ``(# of states)``. The lower diagonal is 
        zero.
        """        
        for i,state in enumerate(self.state_list):
            self.set_state(state)
            for j in self._idx_dict[i]:
                self.A[i,j] = self.trans_prob(self.state_list[j])


    def set_modalities(self, 
                       spsn_dict: Dict[str, List[float]] = {"path": [1., 1.]}):
        """Given specificity :math:`s_P` & sensitivity :math:`s_N` of different 
        diagnostic modalities, compute the system's observation matrix 
        :math:`\\mathbf{B}`.
        """
        self._modality_dict = {}
        for modality, spsn in spsn_dict.items():
            sp, sn = spsn
            self._modality_dict[modality] = np.array([[sp     , 1. - sn],
                                                      [1. - sp, sn     ]])
            
        self._gen_obs_list()
        self._gen_B()


    def _gen_B(self):
        """Generates the observation matrix :math:`\\mathbf{B}`, which contains 
        the :math:`P \\left(Z_t \\mid X_t \\right)`. :math:`\\mathbf{B}` has the 
        shape ``(# of states, # of possible observations)``.
        """
        n_lnl = len(self.lnls)
        self.B = np.zeros(shape=(len(self.state_list), len(self.obs_list)))
        
        for i,state in enumerate(self.state_list):
            self.set_state(state)
            for j,obs in enumerate(self.obs_list):
                diagnoses_dict = {}
                for k,modality in enumerate(self._modality_dict):
                    diagnoses_dict[modality] = obs[n_lnl * k : n_lnl * (k+1)]
                self.B[i,j] = self.obs_prob(diagnoses_dict, log=False)


    def _gen_C(self,
               table: np.ndarray,
               delete_ones: bool = True,
               aggregate_duplicates: bool = True) -> np.ndarray:
        """Generate matrix :math:`\\mathbf{C}` that marginalizes over complete 
        observations when a patient's diagnose is incomplete.
        
        Args:
            table: 2D array where rows represent patients (of the same T-stage) 
                and columns are LNL involvements.
            
            delete_ones: If ``True``, columns in the :math:`\\mathbf{C}` matrix 
                that contain only ones (meaning the respective diagnose is 
                completely unknown) are removed, since they only add zeros to 
                the log-likelihood.
                
            aggregate_duplicates: If ``True``, the number of occurences of 
                diagnoses in the :math:`\\mathbf{C}` matrix is counted and 
                collected in a vector :math:`\\mathbf{f}`. The duplicate 
                columns are then deleted.
        
        Returns:
            Matrix of ones and zeros that can be used to marginalize over 
            possible diagnoses.  
            
        :meta public:     
        """
        C = np.zeros(shape=(len(self.obs_list), len(table)), dtype=bool)
        for i,row in enumerate(table):
            for j,obs in enumerate(self.obs_list):
                # save whether all not missing observations match or not
                C[j,i] = np.all(np.equal(obs, row, 
                                         where=~np.isnan(row.astype(float)),
                                         out=np.ones_like(row, dtype=bool)))
                
        if delete_ones:
            sum_over_C = np.sum(C, axis=0)
            keep_idx = np.argwhere(sum_over_C != len(self.obs_list)).flatten()
            C = C[:,keep_idx]
            
        if aggregate_duplicates:
            C, f = np.unique(C, axis=1, return_counts=True)
            return C, f
            
        return C, np.ones(shape=(len(table)), dtype=int)


    def load_data(
        self,
        data: pd.DataFrame, 
        t_stages: List[int] = [1,2,3,4], 
        spsn_dict: Dict[str, List[float]] = {"path": [1., 1.]}, 
        mode: str = "HMM",
        gen_C_kwargs: dict = {'delete_ones': True, 
                              'aggregate_duplicates': True}
    ):
        """
        Transform tabular patient data (:class:`pd.DataFrame`) into internal 
        representation, consisting of one or several matrices 
        :math:`\\mathbf{C}_{T}` that can marginalize over possible diagnoses.

        Args:
            data: Table with rows of patients. Must have a two-level 
                :class:`MultiIndex` where the top-level has categories 'Info' 
                and the name of the available diagnostic modalities. Under 
                'Info', the second level is only 'T-stage', while under the 
                modality, the names of the diagnosed lymph node levels are 
                given as the columns.

            t_stages: List of T-stages that should be included in the learning 
                process.

            spsn_dict: Dictionary of specificity :math:`s_P` and :math:`s_N` 
                (in that order) for each observational/diagnostic modality.

            mode: ``"HMM"`` for hidden Markov model and ``"BN"`` for Bayesian 
                network.
                
            gen_C_kwargs: Keyword arguments for the :meth:`_gen_C`. For 
                efficiency, both ``delete_ones`` and ``aggregate_duplicates``
                should be set to one, resulting in a smaller :math:`\\mathbf{C}` 
                matrix and an additional count vector :math:`\\mathbf{f}`.
        """
        self.set_modalities(spsn_dict=spsn_dict)
        
        # For the Hidden Markov Model
        if mode=="HMM":
            C_dict = {}
            f_dict = {}

            for stage in t_stages:

                table = data.loc[data[('Info', 'T-stage')] == stage,
                                 self._modality_dict.keys()].values
                C, f = self._gen_C(table, **gen_C_kwargs)
                    
                C_dict[stage] = C.copy()
                f_dict[stage] = f.copy()

            self.C_dict = C_dict
            self.f_dict = f_dict

        # For the Bayesian Network
        elif mode=="BN":
            self.C = np.array([], dtype=int)
            self.f = np.array([], dtype=int)

            table = data[self._modality_dict.keys()].values
            C, f = self._gen_C(table, **gen_C_kwargs)
            
            self.C = C.copy()
            self.f = f.copy()


    def _evolve(self, num_time_steps: int = 10) -> np.ndarray:
        """Evolve hidden Markov model based system over time steps. Compute 
        p(X|t) in matrix form.
        
        Args:
            num_time_steps: Number of time-steps. 
        
        Returns:
            A matrix with the values p(X|t) for each time-step.
        
        :meta public:
        """
        # All healthy state at beginning
        start = np.zeros(shape=len(self.state_list), dtype=float)
        start[0] = 1.
        
        inv_prob = np.zeros(shape=(num_time_steps, len(self.state_list)), 
                            dtype=float)
        
        # loop through time-steps
        for i in range(num_time_steps-1):
            inv_prob[i] = start
            start = start @ self.A
        
        inv_prob[-1] = start
        
        return inv_prob


    def _spread_probs_are_valid(self, theta: np.ndarray) -> bool:
        """Check that the spread probability (rates) are all within limits.
        """
        if theta.shape != self.get_theta().shape:
            msg = ("Shape of provided spread parameters does not match network")
            raise ValueError(msg)
        if np.any(np.greater(0., theta)):
            return False
        if np.any(np.greater(theta, 1.)):
            return False
        
        return True


    def marg_likelihood(
        self, 
        theta: np.ndarray, 
        t_stages: List[int] = ["early", "late"], 
        time_dists: Dict[Any, np.ndarray] = {}, 
        mode: str = "HMM"
    ) -> float:
        """
        Compute the likelihood of the (already stored) data, given the spread 
        parameters, marginalized over time of diagnosis via time distributions.

        Args:
            theta: Set of parameters, consisting of the base probabilities 
                :math:`b` (as many as the system has nodes) and the transition 
                probabilities :math:`t` (as many as the system has edges).

            t_stages: List of T-stages that should be included in the learning 
                process.

            time_dists: Distribution over the probability of diagnosis at 
                different times :math:`t` given T-stage.

            mode: ``"HMM"`` for hidden Markov model and ``"BN"`` for Bayesian 
                network.

        Returns:
            The log-likelihood of the data, given te spread parameters.
        """
        if not self._spread_probs_are_valid(theta):
            return -np.inf

        self.set_theta(theta, mode=mode)

        # likelihood for the hidden Markov model
        if mode == "HMM":
            res = 0
            num_time_steps = len(time_dists[t_stages[0]])
            state_probs = self._evolve(num_time_steps)
            
            for t in t_stages:
                # marginalization using `time_dists`
                p = time_dists[t] @ state_probs @ self.B @ self.C_dict[t]
                res += self.f_dict[t] @ np.log(p)

        # likelihood for the Bayesian network
        elif mode == "BN":
            a = np.ones(shape=(len(self.state_list),), dtype=float)

            for i, state in enumerate(self.state_list):
                self.set_state(state)
                for node in self.lnls:
                    a[i] *= node.bn_prob()

            b = a @ self.B
            res = self.f @ np.log(b @ self.C)

        return res


    def time_likelihood(
        self, 
        theta: np.ndarray, 
        t_stages: List[str] = ["early", "late"],
        num_time_steps: int = 10
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
            
        Returns:
            The likelihood of the data, given the spread parameters as well as 
            the diagnose time for each T-stage.
        """
        # splitting theta into spread parameters and...
        len_spread_params = len(theta) - len(t_stages)
        spread_params = theta[:len_spread_params]
        if not self._spread_probs_are_valid(spread_params):
            return -np.inf
        
        # ...diagnose times for each T-stage
        times = np.around(theta[len_spread_params:]).astype(int)
        if np.any(np.greater(0, times)):
            return -np.inf
        if np.any(np.less(num_time_steps, times)):
            return -np.inf
        
        self.set_theta(spread_params)
        
        res = 0.
        # All healthy state at beginning
        start = np.zeros(shape=len(self.state_list), dtype=float)
        start[0] = 1.

        for i,t in enumerate(t_stages):
            p = start @ mat_pow(self.A, times[i]) @ self.B @ self.C_dict[t]
            res += self.f_dict[t] @ np.log(p)
        
        return res

    
    def risk(
        self,
        theta: Optional[np.ndarray] = None,
        inv: Optional[np.ndarray] = None,
        diagnoses: Dict[str, np.ndarray] = {},
        time_dist: Optional[np.ndarray] = None,
        mode: str = "HMM"
    ) -> Union[float, np.ndarray]:
        """
        Compute risk(s) of involvement given a specific (but potentially 
        incomplete) diagnosis.
        
        Args:
            theta: Set of new spread parameters. If not given (``None``), the 
                currently set parameters will be used.
                
            inv: Specific hidden involvement one is interested in. If only parts 
                of the state are of interest, the remainder can be masked with 
                values ``None``. If specified, the functions returns a single 
                risk.
                
            diagnoses: Dictionary that can hold a potentially incomplete (mask 
                with ``None``) diagnose for every available modality. Leaving 
                out available modalities will assume a completely missing 
                diagnosis.
                
            time_dist: Prior distribution over time. Must sum to 1 and needs 
                to be given for ``mode = "HMM"``.
                
            mode: Set to ``"HMM"`` for the hidden Markov model risk (requires 
                the ``time_dist``) or to ``"BN"`` for the Bayesian network 
                version.
                
        Returns:
            A single probability value if ``inv`` is specified and an array 
            with probabilities for all possible hidden states otherwise.
        """
        # assign theta to system or use the currently set one
        if theta is not None:
            self.set_theta(theta, mode=mode)
            
        # create one large diagnose vector from the individual modalitie's 
        # diagnoses
        obs = np.array([])
        for mod in self._modality_dict:
            if mod in diagnoses:
                obs = np.append(obs, diagnoses[mod])
            else:
                obs = np.append(obs, np.array([None] * len(self.lnls)))
        
        # vector of probabilities of arriving in state x, marginalized over time
        # HMM version
        if mode == "HMM":
            state_probs = self._evolve(len(time_dist))
            pX = time_dist @ state_probs
                
        # BN version
        elif mode == "BN":
            pX = np.ones(shape=(len(self.state_list)), dtype=float)
            for i, state in enumerate(self.state_list):
                self.set_state(state)
                for node in self.lnls:
                    pX[i] *= node.bn_prob()
        
        # compute the probability of observing a diagnose z and being in a 
        # state x which is P(z,x) = P(z|x)P(x). Do that for all combinations of 
        # x and z and put it in a matrix
        pZX = self.B.T * pX
        
        # vector of probabilities for seeing a diagnose z
        pZ = pX @ self.B
        
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
