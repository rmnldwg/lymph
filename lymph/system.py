import numpy as np
import scipy as sp 
import scipy.stats
import pandas as pd
import warnings
from typing import Union, Optional, List, Dict

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
    """Class that describes a whole lymphatic system with its lymph node levels 
    (LNLs) and the connections between them.

    Args:
        graph: For every key in the dictionary, the :class:`system` will 
            create a :class:`node` that represents a binary random variable. The 
            values in the dictionary should then be the a list of names to which 
            :class:`edges` from the current key should be created.
    """
    def __init__(self, 
                 graph: dict = {}):

        self.tumors = []    # list of nodes with type tumour
        self.lnls = []      # list of all lymph node levels
        self.nodes = []     # list of all nodes in the graph
        self.edges = []     # list of all edges connecting nodes in the graph
        
        for key in graph:
            self.nodes.append(Node(key))
            
        for node in self.nodes:
            if node.typ == "tumor":
                self.tumors.append(node)
            else:
                self.lnls.append(node)

        for key, values in graph.items():
            for value in values:
                self.edges.append(Edge(self.find_node(key), 
                                       self.find_node(value)))

        self._gen_state_list()
        self._gen_mask()



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
        """Lists the graph as it was provided when the system was created
        """
        res = []
        for node in self.nodes:
            out = []
            for o in node.out:
                out.append(o.end.name)
            res.append((node.name, out))
            
        return dict(res)



    def print_graph(self):
        """Print info about the structure and parameters of the graph.
        """

        print("Tumor(s):")
        for tumor in self.tumors:
            if tumor.typ != "tumor":
                raise RuntimeError("Tumor node is not of type tumor")
            
            prefix = tumor.name + " ---"
            for o in tumor.out:
                print(f"\t{prefix} {o.t * 100: >4.1f} % --> {o.end.name}")
                prefix = "".join([" "] * (len(tumor.name) + 1)) + "+--"

        print("\nLNL(s):")
        for lnl in self.lnls:
            if lnl.typ != "lnl":
                raise RuntimeError("LNL node is not of type LNL")

            prefix = lnl.name + " ---"
            for o in lnl.out:
                print(f"\t{prefix} {o.t * 100: >4.1f} % --> {o.end.name}")
                prefix = "".join([" "] * (len(lnl.name) + 1)) + "+--"



    def list_edges(self) -> List[Edge]:
        """Lists all edges of the system with its corresponding start and end 
        states
        """
        res = []
        for edge in self.edges:
            res.append((edge.start.name, edge.end.name, edge.t))
            
        return res



    def set_state(self, newstate: List[int]):
        """Sets the state of the system to ``newstate``.
        """
        if len(newstate) != len(self.lnls):
            raise ValueError("length of newstate must match # of LNLs")
        
        for i, node in enumerate(self.lnls):  # only set lnl's states
            node.state = int(newstate[i])



    def get_theta(self) -> List[float]:
        """Returns the parameters currently set. It will return the transition 
        probabilities in the order they appear in the graph dictionary. This 
        deviates somewhat from the notation in the paper, where base and 
        transition probabilities are distinguished as probabilities along edges 
        from primary tumour to LNL and from LNL to LNL respectively.
        """
        theta = np.zeros(shape=(len(self.edges)))
        for i, edge in enumerate(self.edges):
            theta[i] = edge.t

        return theta


    def set_theta(self, 
                  theta: np.ndarray, 
                  mode: str = "HMM"):
        """Fills the system with new base and transition probabilities and also 
        computes the transition matrix A again, if one is in mode "HMM".

        Args:
            theta: The new parameters that should be fed into the system. They 
                all represent the transition probabilities along the edges of 
                the network and will be set in the order they appear in the 
                graph dictionary. As mentioned in the ``get_theta()`` function, 
                this includes the spread probabilities from the primary tumour 
                to the LNLs, as well as the spread among the LNLs.

            mode: If one is in "BN" mode (Bayesian network), then it is not 
                necessary to compute the transition matrix A again, so it is 
                skipped. (default: ``"HMM"``)
        """
        if len(theta) != len(self.edges):
            raise ValueError("# of parameters must match # of edges")
        
        for i, edge in enumerate(self.edges):
            edge.t = theta[i]

        if mode=="HMM":
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
                                        len(self.lnls), n_obs), dtype=int)
        for i in range(2**(n_obs * len(self.lnls))):
            tmp = toStr(i, 2, rev=False, length=n_obs * len(self.lnls))
            for j in range(len(self.lnls)):
                for k in range(n_obs):
                    self.obs_list[i,j,k] = int(tmp[k*len(self.lnls)+j])



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
        self.A = np.zeros(shape=(len(self.state_list), len(self.state_list)))
        for i,state in enumerate(self.state_list):
            self.set_state(state)
            for j in self._idx_dict[i]:
                self.A[i,j] = self.trans_prob(self.state_list[j,:])



    def set_modalities(self, 
                       spsn_dict: Dict[str, List[float]] = {"path": [1., 1.]}):
        """Given some 2x2 matrices for each diagnostic modality based on their 
        specificity and sensitivity, compute observation matrix 
        :math:`\\mathbf{B}` and store the details of the diagnostic modalities.
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
                    


    # -------------------- UNPARAMETRIZED SAMPLING -------------------- #
    def unparametrized_epoch(self, 
                             t_stage: List[int] = [1,2,3,4], 
                             time_prior_dict: dict = {}, 
                             T: float = 1., 
                             scale: float = 1e-2) -> float:
        """An attempt at unparametrized sampling, where the algorithm samples
        A from the full solution space of row-stochastic matrices with zeros
        where transitions are forbidden.

        Args:
            t_stage: List of T-stages that should be included in the learning 
                process. (default: ``[1,2,3,4]``)

            time_prior_dict: Dictionary with keys of T-stages in t_stage and 
                values of time priors for each of those T-stages.

            T: Temperature of the epoch. Can be reduced from a starting value 
                down to almost zero for an annealing approach to sampling.
                (default: ``1.``)

        Returns:
            The log-likelihood of the epoch.
        """
        likely = self.likelihood(t_stage, time_prior_dict)
        
        for i in np.random.permutation(len(self.lnls) -1):
            # Generate Logit-Normal sample around current position
            x_old = self.A[i,self._idx_dict[i]]
            mu = [np.log(x_old[k]) - np.log(x_old[-1]) for k in range(len(x_old)-1)]
            sample = np.random.multivariate_normal(mu, scale*np.eye(len(mu)))
            numerator = 1 + np.sum(np.exp(sample))
            x_new = np.ones_like(x_old)
            x_new[:len(x_old)-1] = np.exp(sample)
            x_new /= numerator

            self.A[i,self._idx_dict[i]] = x_new
            prop_likely = self.likelihood(t_stage, time_prior_dict)

            if np.exp(- (likely - prop_likely) / T) >= np.random.uniform():
                likely = prop_likely
            else:
                self.A[i,self._idx_dict[i]] = x_old

        return likely
    # -------------------- UNPARAMETRIZED END -------------------- #



    def load_data(self,
                  data: pd.DataFrame, 
                  t_stage: List[int] = [1,2,3,4], 
                  spsn_dict: Dict[str, List[float]] = {"path": [1., 1.]}, 
                  mode: str = "HMM"):
        """Generates the matrix C that marginalizes over multiple states for 
        data with incomplete observation, as well as how often these obser-
        vations occur in the dataset. In the end the computation 
        :math:`\\mathbf{p} = \\boldsymbol{\\pi} \\cdot \\mathbf{A}^t \\cdot \\mathbf{B} \\cdot \\mathbf{C}` 
        results in an array of probabilities that can - together with the 
        frequencies :math:`f` - be used to compute the likelihood. This also 
        works for the Bayesian network case: :math:`\\mathbf{p} = \\mathbf{a} \\cdot \\mathbf{C}` 
        where :math:`a` is an array containing the probability for each state.

        Args:
            data: Contains rows of patient data. The columns must include the 
                T-stage and at least one diagnostic modality.

            t_stage: List of T-stages that should be included in the learning 
                process. (default: ``[1,2,3,4]``)

            spsn_dict: Dictionary of specificity :math:`s_P` and :math:`s_N` 
                (in that order) for each observational/diagnostic modality. 
                (default: ``{"path": [1., 1.]}``)

            mode: ``"HMM"`` for hidden Markov model and ``"BN"`` for Bayesian 
                network. (default: ``"HMM"``)
        """
        self.set_modalities(spsn_dict=spsn_dict)
        
        # For the Hidden Markov Model
        if mode=="HMM":
            C_dict = {}
            f_dict = {}

            for stage in t_stage:
                C = np.array([], dtype=int)
                f = np.array([], dtype=int)

                # returns array of patient data that have the same T-stage and 
                # contain diagnoses from those diagnostic modalities that were 
                # specified in the modality_dict argument
                for patient in data.loc[data['Info', 'T-stage'] == stage, 
                                        self._modality_dict.keys()].values:
                    tmp = np.zeros(shape=(len(self.obs_list),1), dtype=int)
                    for i, obs in enumerate(self.obs_list):
                        # returns true if all not missing observations match
                        if np.all(np.equal(obs.flatten(order='F'), 
                                           patient, 
                                           where=~np.isnan(patient), 
                                           out=np.ones_like(patient, 
                                                            dtype=bool))):
                            tmp[i] = 1

                    # build up the matrix C without any duplicates and count 
                    # the occurence of patients with the same pattern in f
                    if len(f) == 0:
                        C = tmp.copy()
                        f = np.append(f, 1)
                    else:
                        found = False
                        for i, col in enumerate(C.T):
                            if np.all(np.equal(col.flatten(), tmp.flatten())):
                                f[i] += 1
                                found = True

                        if not found:
                            C = np.hstack([C, tmp])
                            f = np.append(f, 1)
                            
                # delete columns of the C matrix that contain only ones, meaning 
                # that for that patient, no observation was made
                idx = np.sum(C, axis=0) != len(self.obs_list)
                C = C[:,idx]
                f = f[idx]
                
                C_dict[stage] = C.copy()
                f_dict[stage] = f.copy()

            self.C_dict = C_dict
            self.f_dict = f_dict



        # For the Bayesian Network
        elif mode=="BN":
            self.C = np.array([], dtype=int)
            self.f = np.array([], dtype=int)

            # returns array of pateint data that have the same T-stage
            for patient in data[self._modality_dict.keys()].values:
                tmp = np.zeros(shape=(len(self.obs_list),1), dtype=int)
                for i, obs in enumerate(self.obs_list):
                    # returns true if all observations that aren't missing match
                    if np.all(np.equal(obs.flatten(order='F'), 
                                       patient, 
                                       where=~np.isnan(patient), 
                                       out=np.ones_like(patient, dtype=bool))):
                        tmp[i] = 1

                # build up the matrix C without any duplicates and count the 
                # occurence of patients with the same pattern in f
                if len(self.f) == 0:
                    self.C = tmp.copy()
                    self.f = np.append(self.f, 1)
                else:
                    found = False
                    for i, col in enumerate(self.C.T):
                        if np.all(np.equal(col.flatten(), tmp.flatten())):
                            self.f[i] += 1
                            found = True

                    if not found:
                        self.C = np.hstack([self.C, tmp])
                        self.f = np.append(self.f, 1)

                # delete columns of the C matrix that contain only ones, meaning 
                # that for that patient, no observation was made
                idx = np.sum(self.C, axis=0) != len(self.obs_list)
                self.C = self.C[:,idx]
                self.f = self.f[idx]



    def likelihood(self, 
                   theta: np.ndarray, 
                   t_stage: List[int] = [1,2,3,4], 
                   time_prior_dict: dict = {}, 
                   mode: str = "HMM") -> float:
        """Computes the likelihood of a set of parameters, given the already 
        stored data(set).

        Args:
            theta: Set of parameters, consisting of the base probabilities 
                :math:`b` (as many as the system has nodes) and the transition 
                probabilities :math:`t` (as many as the system has edges).

            t_stage: List of T-stages that should be included in the learning 
                process. (default: ``[1,2,3,4]``)

            time_prior_dict: Dictionary with keys of T-stages in ``t_stage`` and 
                values of time priors for each of those T-stages.

            mode: ``"HMM"`` for hidden Markov model and ``"BN"`` for Bayesian 
                network. (default: ``"HMM"``)

        Returns:
            The log-likelihood of a parameter sample.
        """
        # check if all parameters are within their limits
        if np.any(np.greater(0., theta)):
            return -np.inf
        if np.any(np.greater(theta, 1.)):
            return -np.inf

        self.set_theta(theta, mode=mode)

        # likelihood for the hidden Markov model
        if mode == "HMM":
            res = 0
            for stage in t_stage:

                start = np.zeros(shape=(len(self.state_list),))
                start[0] = 1.
                tmp = np.zeros(shape=(len(self.state_list),))

                for pt in time_prior_dict[stage]:
                    tmp += pt * start
                    start = start @ self.A

                p = tmp @ self.B @ self.C_dict[stage]
                res += self.f_dict[stage] @ np.log(p)


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



    # -------------------------- SPECIAL LIKELIHOOD -------------------------- #
    def beta_likelihood(self, theta, beta, t_stage=[1, 2, 3, 4], time_prior_dict={}):
        if np.any(np.greater(0., theta)):
            return -np.inf, -np.inf
        if np.any(np.greater(theta, 1.)):
            return -np.inf, -np.inf

        bn = self.likelihood(theta, mode="BN")
        hmm = self.likelihood(theta, t_stage, time_prior_dict, mode="HMM")

        res = beta * bn + (1-beta) * hmm
        diff_q = bn - hmm
        return res, diff_q



    def binom_llh(self, p, t_stage=["late"], T_max=10):

        if np.any(np.greater(0., p)) or np.any(np.greater(p, 1.)):
            return -np.inf

        t = np.asarray(range(1,T_max+1))
        pt = lambda p : sp.stats.binom.pmf(t-1,T_max,p)
        time_prior_dict = {}

        for i, stage in enumerate(t_stage):
            time_prior_dict[stage] = pt(p[i])

        return self.likelihood(self.get_theta(), t_stage, time_prior_dict, mode="HMM")



    def combined_likelihood(self, 
                            theta: np.ndarray, 
                            t_stage: List[str] = ["early", "late"], 
                            time_prior_dict: dict = {}, 
                            T_max: int = 10) -> float:
        """Likelihood for learning both the system's parameters and the center 
        of a Binomially shaped time prior.
        
        Args:
            theta: Set of parameters, consisting of the base probabilities 
                :math:`b` (as many as the system has nodes), the transition 
                probabilities :math:`t` (as many as the system has edges) and - 
                in this particular case - the binomial parameters for all but 
                the first T-stage's time prior.
                
            t_stage: keywords of T-stages that are present in the dictionary of 
                C matrices. (default: ``["early", "late"]``)
                
            time_prior_dict: Dictionary with keys of T-stages in ``t_stage`` and 
                values of time priors for each of those T-stages.
                
            T_max: maximum number of time steps. TODO: make this more consistent
            
        Returns:
            The combined likelihood of observing patients with different 
            T-stages, given the spread probabilities as well as the parameters 
            for the later (except the first) T-stage's binomial time prior.
        """

        if np.any(np.greater(0., theta)) or np.any(np.greater(theta, 1.)):
            return -np.inf

        theta, p = theta[:-1], theta[-1]
        t = np.asarray(range(1,T_max+1))
        pt = lambda p : sp.stats.binom.pmf(t-1,T_max,p)

        time_prior_dict["early"] = pt(0.4)
        time_prior_dict["late"] = pt(p)
        
        return self.likelihood(theta, t_stage, time_prior_dict, mode="HMM")
    # ----------------------------- SPECIAL DONE ----------------------------- #



    def risk(self, 
             inv: np.ndarray, 
             obs: Dict[np.ndarray], 
             time_prior: List[float] = [], 
             mode: str = "HMM") -> float:
        """Computes the risk for involvement (or no involvement), given some 
        observations and a time distribution for the Markov model (and the 
        Bayesian network).

        Args:
            inv: Pattern of involvement that we want to compute the risk for. 
                Values can take on the values ``0`` (*negative*), ``1`` 
                (*positive*) and ``None`` of we don't care if this is involved 
                or not.

            obs: Holds a diagnose of similar kind as ``inv`` for each diagnostic 
                modality. An incomplete diagnose can be filled with ``None``.

            time_prior: Discrete distribution over the time steps. Must hence 
                sum to 1.

            mode: ``"HMM"`` for hidden Markov model and ``"BN"`` for Bayesian 
                network. (default: ``"HMM"``)

        Returns:
            The risk for the involvement of interest, given an observation.
        """
        if len(inv) != len(self.lnls):
            raise ValueError("The involvement array has the wrong length. "
                             f"It should be {len(self.lnls)}")
            
        # construct a diagnose in a single numpy array (in the way it is stored 
        # in the obs_list) from the dictionary of diagnoses.
        n_obs = len(self._modality_dict)
        full_obs = np.array([None] * len(self.lnls) * n_obs)
        emptyobs = True
        for i, modality in enumerate(self._modality_dict):
            if modality in obs:
                emptyobs = False
                full_obs[i * n_obs : (i+1) * n_obs] = obs[modality]

        if emptyobs:
            raise ValueError("Provided diagnose does not contain any specified "
                             "diagnostic modalities.")

        # P(Z), probability of observing a certain (observational) state
        pZ = np.zeros(shape=(len(self.obs_list),))
        start = np.zeros(shape=(len(self.state_list),))
        start[0] = 1.

        # in this case, HMM and BN only differ in the way this pX is computed
        # here HMM
        if mode == "HMM":
            # P(X), probability of arriving at a certain (hidden) state
            pX = np.zeros(shape=(len(self.state_list),))
            for pt in time_prior:
                pX += pt * start
                start = start @ self.A

        # here BN
        elif mode == "BN":
            # P(X), probability of a certain (hidden) state
            pX = np.ones(shape=(len(self.state_list),))
            for i, state in enumerate(self.state_list):
                self.set_state(state)
                for node in self.lnls:
                    pX[i] *= node.bn_prob()

        pZ = pX @ self.B

        # figuring out which states I should sum over, given some node levels
        idxX = np.array([], dtype=int)
        idxZ = np.array([], dtype=int)

        for i, x in enumerate(self.state_list):
            if np.all(np.equal(x,inv, 
                               where=(inv!=None), 
                               out=np.ones_like(inv, dtype=bool))):
                idxX = np.append(idxX, i)
        
        # loop over all possible complete observations...
        for i, z in enumerate(self.obs_list):
            # ...and check whether they could match the given diagnose
            if np.all(np.equal(z, full_obs, 
                               where=(full_obs!=None), 
                               out=np.ones_like(full_obs, dtype=bool))):
                idxZ = np.append(idxZ, i)

        num = 0.
        denom = 0.

        for iz in idxZ:
            # evidence, marginalizing over all states that share the 
            # involvement of interest
            denom += pZ[iz]
            for ix in idxX:
                # likelihood times prior, marginalizing over all states that 
                # share the involvement of interest
                num += self.B[ix,iz] * pX[ix]

        risk = num / denom
        return risk
