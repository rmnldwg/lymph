import numpy as np
import scipy as sp 
import scipy.stats
import warnings

from .node import Node
from .edge import Edge

def toStr(n, base, rev=False, length=None):
    """Function that converts an integer into another base.
    
    Args:
        n (int): Number to convert

        base (int): Base of the resulting converted number

        rev (bool): If true, the converted number will be printed in reverse 
            order.
            (default: ``False``)

        length (int or None): Length of the returned string.
            (default: ``None``)

    This function returns the (padded) string of the converted number.
    """
    
    if base > 16:
        raise Exception("Base must be 16 or smaller!")
        
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
        graph (dict): For every key in the dictionary, the :class:`system` will 
            create a :class:`node` that represents a binary random variable. The 
            values in the dictionary should then be the a list of names to which 
            :class:`edges` from the current key should be created.

        obs_table (numpy array, 3D): A 2D arrray for each observational modality.
            These 2D arrays contain the conditional probabilities of observations 
            given hidden states (like sensitivity :math:`s_N` and specificity 
            :math:`s_P`). Each of the 2D arrays must be "column-stochastic".
    """
    def __init__(self, graph={}, obs_table=np.array([[[1. , 0. ], 
                                                      [0. , 1. ]], 
                                                     [[0.8, 0.2], 
                                                      [0.2, 0.8]]])):
        self.narity = obs_table.shape[::-1][0]
        self.n_obs = obs_table.shape[0]
        self.nodes = []
        self.edges = []
        for key in graph:
            self.nodes.append(Node(key, obs_table=obs_table))

        for key, values in graph.items():
            for value in values:
                self.edges.append(Edge(self.find_node(key), 
                                       self.find_node(value), 
                                       narity=self.narity))

        self.gen_state_list()
        self.gen_obs_list()
        self.gen_mask()
        self.gen_B()



    def find_node(self, name):
        """Finds and returns a node with name ``name``.
        """
        for node in self.nodes:
            if node.name == name:
                return node



    def find_edge(self, startname, endname):
        """Finds and returns the edge instance which has a parent node named 
        ``startname`` and ends with node ``endname``.
        """
        for node in self.nodes:
            if node.name == startname:
                for o in node.out:
                    if o.end.name == endname:
                        return o



    def list_graph(self):
        """Lists the graph as it was provided when the system was created
        """
        res = []
        for node in self.nodes:
            out = []
            for o in node.out:
                out.append(o.end.name)
            res.append((node.name, out))
        return dict(res)



    def list_edges(self):
        """Lists all edges of the system with its corresponding start and end states
        """
        res = []
        for edge in self.edges:
            res.append((edge.start.name, edge.end.name, edge.t))
        return res



    def set_state(self, newstate):
        """Sets the state of the system to ``newstate``.
        """
        for i, node in enumerate(self.nodes):
            node.state = newstate[i]



    def get_theta(self):
        """Returns the parameters currently set. Note that this will include 
        the evolution probability parameters that are currently not in use 
        (will hence be set to zero). So it might return an array of unexpected 
        size.
        """
        theta = np.zeros(shape=(2*len(self.nodes) + (self.narity-1) * len(self.edges),), 
                         dtype=float)
        for i, node in enumerate(self.nodes):
            theta[(self.narity-1)*i] = node.p
            theta[(self.narity-1)*i+1] = node.ep
        for i, edge in enumerate(self.edges):
            for j in range(1, self.narity):
                theta[(self.narity-1)*len(self.nodes) + i*(self.narity-1) + (j-1)] = edge.t[j]
        return theta



    def set_theta(self, theta, mode="HMM"):
        """Fills the system with new base and transition probabilities and also 
        computes the transition matrix A again, if one is in mode "HMM".

        Args:
            theta (numpy array): The new parameters that should be fed into the 
                system. The first :math:`2N` numbers in the array contain the 
                base probabilities :math:`b`, and evolution probabilities 
                :math:`t` (if narity = 3), where :math:`N` is the number of 
                nodes. The remaining entries contain either one or two 
                parameters for each edge in the system (depending on the narity).

            mode (str): If one is in "BN" mode (Bayesian network), then it is 
                not necessary to compute the transition matrix A again, so it is 
                skipped.
                (default: ``"HMM"``)
        """
        for i, node in enumerate(self.nodes):
            node.p = theta[i]
            node.ep = theta[i+1]
        for i, edge in enumerate(self.edges):
            edge.t[0] = 0.0
            for j in range(1, self.narity):
                edge.t[j] = theta[len(self.nodes) + i + (j-1)]

        # I'm not gonna use that narity stuff for a while...
        # for i, node in enumerate(self.nodes):
        #     node.p = theta[(self.narity-1)*i]
        #     node.ep = theta[(self.narity-1)*i+1]
        # for i, edge in enumerate(self.edges):
        #     edge.t[0] = 0.0
        #     for j in range(1, self.narity):
        #         edge.t[j] = theta[(self.narity-1)*len(self.nodes) + i*(self.narity-1) + (j-1)]

        if mode=="HMM":
            self.gen_A()



    def trans_prob(self, newstate, log=False, acquire=False):
        """Computes the probability to transition to newstate, given its 
        current state.

        Args:
            newstate (list): List of new states for each node in the lymphatic 
                system. The transition probability :math:`t` will be computed 
                from the current states to these states.

            log (bool): if ``True``, the log-probability is computed.
                (default: ``False``)

            acquire (bool): if ``True``, after computing and returning the 
                probability, the system updates its own state to be ``newstate``.
                (default: ``False``)

        This method returns the transition probability :math:`t`.
        """
        if not log:
            res = 1
            for i, node in enumerate(self.nodes):
                res *= node.trans_prob(log=log)[newstate[i]]
        else:
            res = 0
            for i, node in enumerate(self.nodes):
                res += node.trans_prob(log=log)[newstate[i]]

        if acquire:
            self.set_state(newstate)

        return res



    def obs_prob(self, observation, log=False):
        """Computes the probability to see a certain observation, given the 
        system's current state.

        Args:
            observation (numpy array, shape=(n_obs, len(self.nodes))): Contains 
                the observed state of the patient. if there are :math:`N` 
                different diagnosing modalities and :math:`M` nodes this has the 
                shape :math:`(N,M)` and contains zeros and ones.

            log (bool): If True, this returns the log probability.
                (default: ``False``)

        This method returns the probability to see the given observation.
        """
        if not log:
            res = 1
            for i, node in enumerate(self.nodes):
                res *= node.obs_prob(log=log, observation=observation[i])
        else:
            res = 0
            for i, node in enumerate(self.nodes):
                res += node.obs_prob(log=log, observation=observation[i])

        return res



    def gen_state_list(self):
        """Generates the list of (hidden) states.
        """
        self.state_list = np.zeros(shape=(self.narity**len(self.nodes), 
                                          len(self.nodes)), dtype=int)
        for i in range(self.narity**len(self.nodes)):
            tmp = toStr(i, self.narity, rev=False, length=len(self.nodes))
            for j in range(len(self.nodes)):
                self.state_list[i,j] = int(tmp[j])



    def gen_obs_list(self):
        """Generates the list of possible observations.
        """
        self.obs_list = np.zeros(shape=(2**(self.n_obs * len(self.nodes)), 
                                        len(self.nodes), self.n_obs), dtype=int)
        for i in range(2**(self.n_obs * len(self.nodes))):
            tmp = toStr(i, 2, rev=False, length=self.n_obs * len(self.nodes))
            for j in range(len(self.nodes)):
                for k in range(self.n_obs):
                    self.obs_list[i,j,k] = int(tmp[k*len(self.nodes)+j])



    def gen_mask(self):
        """Generates a dictionary that contains for each row of :math:`\\mathbf{A}` 
        the indices where :math:`\\mathbf{A}` is NOT zero.
        """
        self.idx_dict = {}
        for i in range(self.narity**len(self.nodes)):
            self.idx_dict[i] = []
            for j in range(self.narity**len(self.nodes)):
                if not np.any(np.greater(self.state_list[i,:], 
                                         self.state_list[j,:])):
                    self.idx_dict[i].append(j)



    def gen_A(self):
        """Generates the transition matrix :math:`\\mathbf{A}`, which contains the 
        :math:`P \\left( X_{t+1} \\mid X_t \\right)`. :math:`\\mathbf{A}` is a 
        square matrix with size ``(# of states)``. The lower diagonal is zero.
        """
        self.A = np.zeros(shape=(self.narity**len(self.nodes), self.narity**len(self.nodes)))
        for i in range(self.narity**len(self.nodes)):
            self.set_state(self.state_list[i,:])
            for j in self.idx_dict[i]:
                self.A[i,j] = self.trans_prob(self.state_list[j,:])



    def gen_B(self):
        """Generates the observation matrix :math:`\\mathbf{B}`, which contains 
        the :math:`P \\left(Z_t \\mid X_t \\right)`. :math:`\\mathbf{B}` has the 
        shape ``(# of states, # of possible observations)``.
        """
        self.B = np.zeros(shape=(self.narity**len(self.nodes), 2**(self.n_obs * len(self.nodes))))
        for i in range(self.narity**len(self.nodes)):
            self.set_state(self.state_list[i,:])
            for j in range(2**(self.n_obs * len(self.nodes))):
                self.B[i,j] = self.obs_prob(self.obs_list[j])


    # -------------------- UNPARAMETRIZED SAMPLING -------------------- #
    def unparametrized_epoch(self, t_stage=[1,2,3,4], time_dist_dict={}, T=1, 
                             scale=1e-2):
        """An attempt at unparametrized sampling, where the algorithm samples
        A from the full solution space of row-stochastic matrices with zeros
        where transitions are forbidden.

        Args:
            t_stage (list): List of T-stages that should be included in the 
                learning process.
                (default: ``[1,2,3,4]``)

            time_dist_dict (dict): Dictionary with keys of T-stages in t_stage 
                and values of time priors for each of those T-stages.

            T (float): Temperature of the epoch. Can be reduced from a starting 
                value down to almost zero for an annealing approach to sampling.
                (default: ``1``)

        This method returns the log-likelihood of the epoch.
        """
        likely = self.likelihood(t_stage, time_dist_dict)
        for i in np.random.permutation(self.narity**len(self.nodes) -1):
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



    def gen_C(self, data, t_stage=[1,2,3,4], observations=['pCT', 'path'], 
              mode="HMM"):
        """Generates the matrix C that marginalizes over multiple states for 
        data with incomplete observation, as well as how often these obser-
        vations occur in the dataset. In the end the computation 
        :math:`\\mathbf{p} = \\boldsymbol{\\pi} \\cdot \\mathbf{A}^t \\cdot \\mathbf{B} \\cdot \\mathbf{C}` 
        results in an array of probabilities that can - together with the 
        frequencies :math:`f` - be used to compute the likelihood. This also 
        works for the Bayesian network case: :math:`\\mathbf{p} = \\mathbf{a} \\cdot \\mathbf{C}` 
        where :math:`a` is an array containing the probability for each state.

        Args:
            data (pandas dataframe): Contains rows of patient data. The columns 
                must include the T-stage and at least one diagnostic modality.

            t_stage (list): List of T-stages that should be included in the 
                learning process.
                (default: ``[1,2,3,4]``)

            observations (list of str): List of observational modalities from 
                the pandas dataframe that should be included.
                (default: ``['pCT', 'path']``)

            mode (str): ``"HMM"`` for hidden Markov model and ``"BN"`` for Bayesian network.
                (default: ``"HMM"``)
        """

        # For the Hidden Markov Model
        if mode=="HMM":
            C_dict = {}
            f_dict = {}

            for stage in t_stage:
                C = np.array([], dtype=int)
                f = np.array([], dtype=int)

                # returns array of pateint data that have the same T-stage
                for patient in data.loc[data['Info', 'T-stage'] == stage, observations].values:
                    tmp = np.zeros(shape=(len(self.obs_list),1), dtype=int)
                    for i, obs in enumerate(self.obs_list):
                        # returns true if all not missing observations match
                        if np.all(np.equal(obs.flatten(order='F'), 
                                           patient, 
                                           where=~np.isnan(patient), 
                                           out=np.ones_like(patient, dtype=bool))):
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
                    # returns true if all observations that are not missing match
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



    def likelihood(self, theta, t_stage=[1,2,3,4], time_dist_dict={}, mode="HMM"):
        """Computes the likelihood of a set of parameters, given the already 
        stored data(set).

        Args:
            theta (numpy array): Set of parameters, consisting of the base 
                probabilities :math:`b` (as many as the system has nodes), the 
                transition probabilities :math:`t` (as many as the system has 
                edges) and - not yet implemented - the evolution probabilities 
                :math:`e` (as many as the system has nodes).

            t_stage (list): List of T-stages that should be included in the 
                learning process.
                (default: ``[1,2,3,4]``)

            time_dist_dict (dict): Dictionary with keys of T-stages in ``t_stage`` 
                and values of time priors for each of those T-stages.

            mode (str): ``"HMM"`` for hidden Markov model and ``"BN"`` for 
                Bayesian network.
                (default: ``"HMM"``)

        This method returns the log-likelihood of a parameter sample.
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

            if self.narity == 2:
                for i, state in enumerate(self.state_list):
                    self.set_state(state)
                    for node in self.nodes:
                        a[i] *= node.bn_prob()

            b = a @ self.B
            res = self.f @ np.log(b @ self.C)

        return res


    # -------------------- SPECIAL LIKELIHOODS -------------------- #
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



    def combined_likelihood(self, theta, t_stage=['sanguineti'], time_dist_dict={}, T_max=10):
        """Likelihood for learning both the system's parameters and the center of
        a Binomially shaped time prior.
        """

        if np.any(np.greater(0., theta)) or np.any(np.greater(theta, 1.)):
            return -np.inf

        theta, p = theta[:7], theta[7:]
        t = np.asarray(range(1,T_max+1))
        pt = lambda p : sp.stats.binom.pmf(t-1,T_max,p)

        time_dist_dict["early"] = pt(0.4)

        for i, stage in enumerate(t_stage[1:]):
            time_dist_dict[stage] = pt(p[i])

        return self.likelihood(theta, t_stage, time_dist_dict, mode="HMM")
    # -------------------- SPECIAL END -------------------- #


    def risk(self, inv, obs, time_dist=[], mode="HMM"):
        """Computes the risk for involvement (or no involvement), given some 
        observations and a time distribution for the Markov model (and the 
        Bayesian network).

        Args:
            inv (numpy array): Contains None 
                for node levels that one is not interested in or one of the 
                possible states for the respective involvement.
                ``shape=(len(self.state_list))``

            obs (numpy array): Contains ``None`` for 
                node levels where no observation is available and 0 or 1 for the 
                respective observation.
                ``shape=(len(self.obs_list))``

            mode (str): ``"HMM"`` for hidden Markov model and ``"BN"`` for 
                Bayesian network.
                (default: ``"HMM"``)

        This method returns the risk for the involvement of interest, given an 
        observation.
        """
        if len(inv) != len(self.nodes):
            raise Exception("The involvement array has the wrong length." + 
                            "It should be {}".format(len(self.nodes)))

        if len(obs) != len(self.nodes * (self.n_obs)):
            raise Exception("The observation array has the wrong length." + 
                            "It should be {}".format(len(self.nodes * self.n_obs)))

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
            if self.narity == 2:
                for i, state in enumerate(self.state_list):
                    self.set_state(state)
                    for node in self.nodes:
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