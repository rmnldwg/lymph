import numpy as np
import scipy as sp 
import scipy.stats
import pandas as pd
import warnings
from typing import Union, Optional, List

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

        obs_table: An arrray of 2x2 matrices. Each of those matrices represents 
            an observational modality. The (0,0) entry corresponds to the 
            specificity :math:`s_P` and at (1,1) one finds the sensitivity 
            :math:`s_N`. These matrices must be column-stochastic.
    """
    def __init__(self, graph: dict = {}, 
                 obs_table: np.ndarray = np.array([[[1. , 0. ], 
                                                    [0. , 1. ]]])):

        self.n_obs = obs_table.shape[0]
        self.tumors = []   # list of nodes with type tumour
        self.lnls = []      # list of all lymph node levels
        self.nodes = []     # list of all nodes in the graph
        self.edges = []     # list of all edges connecting nodes in the graph
        
        for key in graph:
            self.nodes.append(Node(key, obs_table=obs_table))
            
        for node in self.nodes:
            if node.typ == "tumor":
                self.tumors.append(node)
            else:
                self.lnls.append(node)

        for key, values in graph.items():
            for value in values:
                self.edges.append(Edge(self.find_node(key), 
                                       self.find_node(value)))

        self.gen_state_list()
        self.gen_obs_list()
        self.gen_mask()
        self.gen_B()



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

            for o in tumor.out:
                print(f"{tumor.name} -- {o.t * 100: >4.1f} % --> {o.end.name}")

        print("\nLNL(s):")
        for lnl in self.lnls:
            if lnl.typ != "lnl":
                raise RuntimeError("LNL node is not of type LNL")

            for o in lnl.out:
                print(f"{lnl.name} -- {o.t * 100: >4.1f} % --> {o.end.name}")



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
            self.gen_A()



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
                 observation: np.ndarray, 
                 log: bool = False) -> float:
        """Computes the probability to see a certain observation, given the 
        system's current state.

        Args:
            observation: Contains the observed state of the patient. if there 
                are :math:`N` different diagnosing modalities and :math:`M` 
                nodes this has the shape :math:`(N,M)` and contains zeros and 
                ones.

            log: If ``True``, the log probability is computed. 
                (default: ``False``)

        Returns:
            The probability to see the given observation.
        """
        if len(observation) != len(self.lnls):
            raise ValueError("length of observation must match # of LNLs")
        
        if not log:
            res = 1.
            for i, lnl in enumerate(self.lnls):
                res *= lnl.obs_prob(log=log, observation=observation[i])
        else:
            res = 0.
            for i, lnl in enumerate(self.lnls):
                res += lnl.obs_prob(log=log, observation=observation[i])

        return res



    def gen_state_list(self):
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



    def gen_obs_list(self):
        """Generates the list of possible observations.
        """
        self.obs_list = np.zeros(shape=(2**(self.n_obs * len(self.lnls)), 
                                        len(self.lnls), self.n_obs), dtype=int)
        for i in range(2**(self.n_obs * len(self.lnls))):
            tmp = toStr(i, 2, rev=False, length=self.n_obs * len(self.lnls))
            for j in range(len(self.lnls)):
                for k in range(self.n_obs):
                    self.obs_list[i,j,k] = int(tmp[k*len(self.lnls)+j])



    def gen_mask(self):
        """Generates a dictionary that contains for each row of 
        :math:`\\mathbf{A}` those indices where :math:`\\mathbf{A}` is NOT zero.
        """
        self.idx_dict = {}
        for i in range(len(self.state_list)):
            self.idx_dict[i] = []
            for j in range(len(self.state_list)):
                if not np.any(np.greater(self.state_list[i,:], 
                                         self.state_list[j,:])):
                    self.idx_dict[i].append(j)



    def gen_A(self):
        """Generates the transition matrix :math:`\\mathbf{A}`, which contains 
        the :math:`P \\left( X_{t+1} \\mid X_t \\right)`. :math:`\\mathbf{A}` 
        is a square matrix with size ``(# of states)``. The lower diagonal is 
        zero.
        """
        self.A = np.zeros(shape=(len(self.state_list), len(self.state_list)))
        for i,state in enumerate(self.state_list):
            self.set_state(state)
            for j in self.idx_dict[i]:
                self.A[i,j] = self.trans_prob(self.state_list[j,:])



    def gen_B(self):
        """Generates the observation matrix :math:`\\mathbf{B}`, which contains 
        the :math:`P \\left(Z_t \\mid X_t \\right)`. :math:`\\mathbf{B}` has the 
        shape ``(# of states, # of possible observations)``.
        """
        self.B = np.zeros(shape=(len(self.state_list), len(self.obs_list)))
        for i,state in enumerate(self.state_list):
            self.set_state(state)
            for j,obs in enumerate(self.obs_list):
                self.B[i,j] = self.obs_prob(obs)


    # -------------------- UNPARAMETRIZED SAMPLING -------------------- #
    def unparametrized_epoch(self, 
                             t_stage: List[int] = [1,2,3,4], 
                             time_dist_dict: dict = {}, 
                             T: float = 1., 
                             scale: float = 1e-2) -> float:
        """An attempt at unparametrized sampling, where the algorithm samples
        A from the full solution space of row-stochastic matrices with zeros
        where transitions are forbidden.

        Args:
            t_stage: List of T-stages that should be included in the learning 
                process. (default: ``[1,2,3,4]``)

            time_dist_dict: Dictionary with keys of T-stages in t_stage and 
                values of time priors for each of those T-stages.

            T: Temperature of the epoch. Can be reduced from a starting value 
                down to almost zero for an annealing approach to sampling.
                (default: ``1.``)

        Returns:
            The log-likelihood of the epoch.
        """
        likely = self.likelihood(t_stage, time_dist_dict)
        
        for i in np.random.permutation(len(self.lnls) -1):
            # Generate Logit-Normal sample around current position
            x_old = self.A[i,self.idx_dict[i]]
            mu = [np.log(x_old[k]) - np.log(x_old[-1]) for k in range(len(x_old)-1)]
            sample = np.random.multivariate_normal(mu, scale*np.eye(len(mu)))
            numerator = 1 + np.sum(np.exp(sample))
            x_new = np.ones_like(x_old)
            x_new[:len(x_old)-1] = np.exp(sample)
            x_new /= numerator

            self.A[i,self.idx_dict[i]] = x_new
            prop_likely = self.likelihood(t_stage, time_dist_dict)

            if np.exp(- (likely - prop_likely) / T) >= np.random.uniform():
                likely = prop_likely
            else:
                self.A[i,self.idx_dict[i]] = x_old

        return likely
    # -------------------- UNPARAMETRIZED END -------------------- #



    def gen_C(self, 
              data: pd.DataFrame, 
              t_stage: List[int] = [1,2,3,4], 
              observations: List[str] = ['pCT', 'path'], 
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

            observations: List of observational modalities from the pandas 
                :class:`DataFrame` that should be included. 
                (default: ``['pCT', 'path']``)

            mode: ``"HMM"`` for hidden Markov model and ``"BN"`` for Bayesian 
                network. (default: ``"HMM"``)
        """

        # For the Hidden Markov Model
        if mode=="HMM":
            C_dict = {}
            f_dict = {}

            for stage in t_stage:
                C = np.array([], dtype=int)
                f = np.array([], dtype=int)

                # returns array of pateint data that have the same T-stage
                for patient in data.loc[data['Info', 'T-stage'] == stage, 
                                        observations].values:
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
            for patient in data[observations].values:
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
                   time_dist_dict: dict = {}, 
                   mode: str = "HMM") -> float:
        """Computes the likelihood of a set of parameters, given the already 
        stored data(set).

        Args:
            theta: Set of parameters, consisting of the base probabilities 
                :math:`b` (as many as the system has nodes) and the transition 
                probabilities :math:`t` (as many as the system has edges).

            t_stage: List of T-stages that should be included in the learning 
                process. (default: ``[1,2,3,4]``)

            time_dist_dict: Dictionary with keys of T-stages in ``t_stage`` and 
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

                for pt in time_dist_dict[stage]:
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
    def beta_likelihood(self, theta, beta, t_stage=[1, 2, 3, 4], time_dist_dict={}):
        if np.any(np.greater(0., theta)):
            return -np.inf, -np.inf
        if np.any(np.greater(theta, 1.)):
            return -np.inf, -np.inf

        bn = self.likelihood(theta, mode="BN")
        hmm = self.likelihood(theta, t_stage, time_dist_dict, mode="HMM")

        res = beta * bn + (1-beta) * hmm
        diff_q = bn - hmm
        return res, diff_q



    def binom_llh(self, p, t_stage=["late"], T_max=10):

        if np.any(np.greater(0., p)) or np.any(np.greater(p, 1.)):
            return -np.inf

        t = np.asarray(range(1,T_max+1))
        pt = lambda p : sp.stats.binom.pmf(t-1,T_max,p)
        time_dist_dict = {}

        for i, stage in enumerate(t_stage):
            time_dist_dict[stage] = pt(p[i])

        return self.likelihood(self.get_theta(), t_stage, time_dist_dict, mode="HMM")



    def combined_likelihood(self, 
                            theta: np.ndarray, 
                            t_stage: List[str] = ["early", "late"], 
                            time_dist_dict: dict = {}, 
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
                
            time_dist_dict: Dictionary with keys of T-stages in ``t_stage`` and 
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

        time_dist_dict["early"] = pt(0.4)
        time_dist_dict["late"] = pt(p)
        
        return self.likelihood(theta, t_stage, time_dist_dict, mode="HMM")
    # ----------------------------- SPECIAL DONE ----------------------------- #



    def risk(self, 
             inv: np.ndarray, 
             obs: np.ndarray, 
             time_dist: List[float] = [], 
             mode: str = "HMM") -> float:
        """Computes the risk for involvement (or no involvement), given some 
        observations and a time distribution for the Markov model (and the 
        Bayesian network).

        Args:
            inv: Contains ``None`` for node levels that one is not interested 
                in or one of the possible states for the respective involvement.

            obs: Contains ``None`` for node levels where no observation is 
                available and 0 or 1 for the respective observation.

            mode: ``"HMM"`` for hidden Markov model and ``"BN"`` for Bayesian 
                network. (default: ``"HMM"``)

        Returns:
            The risk for the involvement of interest, given an observation.
        """
        if len(inv) != len(self.lnls):
            raise ValueError("The involvement array has the wrong length. "
                             f"It should be {len(self.lnls)}")

        if len(obs) != len(self.lnls * (self.n_obs)):
            raise ValueError("The observation array has the wrong length. "
                             f"It should be {len(self.lnls * self.n_obs)}")

        # P(Z), probability of observing a certain (observational) state
        pZ = np.zeros(shape=(len(self.obs_list),))
        start = np.zeros(shape=(len(self.state_list),))
        start[0] = 1.

        # in this case, HMM and BN only differ in the way this pX is computed
        # here HMM
        if mode == "HMM":
            # P(X), probability of arriving at a certain (hidden) state
            pX = np.zeros(shape=(len(self.state_list),))
            for pt in time_dist:
                pX += pt * start
                start = start  @ self.A

        # here BN
        elif mode == "BN":
            # P(X), probability of a certain (hidden) state
            pX = np.ones(shape=(len(self.state_list),))
            for i, state in enumerate(self.state_list):
                self.set_state(state)
                for node in self.lnls:
                    pX[i] *= node.bn_prob()

        pZ = pX @ self.B

        idxX = np.array([], dtype=int)
        idxZ = np.array([], dtype=int)

        # figuring out which states I should sum over, given some node levels
        for i, x in enumerate(self.state_list):
            if np.all(np.equal(x,inv, 
                               where=inv!=None, 
                               out=np.ones_like(inv, dtype=bool))):
                idxX = np.append(idxX, i)
         
        for i, z in enumerate(self.obs_list):
            if np.all(np.equal(z.flatten(order='F'), 
                               obs, 
                               where=obs!=None, 
                               out=np.ones_like(obs, dtype=bool))):
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