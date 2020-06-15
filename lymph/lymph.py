import numpy as np
import scipy as sp
import scipy.stats


def toStr(n, base, rev=False, length=None):
    """Function that converts an integer into another base.
    
    Parameters
    ----------
    n : int
        Number to convert

    base : int
        Base of the resulting converted number

    rev : bool, default=False
        If true, the converted number will be printed in reverse order

    length : int or None, default=None
        Length of the returned string.

    Returns
    -------
    result : str
        (Padded) string of the converted number"""
    
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

        

class Node:
    """Class for lymph node levels (LNLs) in a lymphatic system.

    Attributes
    ----------
    name : str
        Name of the node

    state : int
        Current state this LNL is in. Can be in {0, ..., narity-1}

    p : float
        Base probability. Probability that this LNL gets infected from the 
        primary tumour.

    ep : float
        Evolution probability. In case of narity = 3 this is the probability
        that the LNL spontaneously develops from state = 1 into state = 2.

    inc : list of Node instances
        Incoming edges from parent nodes

    out : list of Node instances
        Outgoing edges to child nodes

    narity :  int
        Number of different states this LNL can take on

    n_obs : int
        Number of observational methods that are connected to this node

    obs_table : np.array
        Kind of conditional probability table for the observational modalities.
        Contains sensitivity and specificity for each observation method.
    """
    def __init__(self, name, state=0, p=0.0, ep=0.0, 
                 obs_table=np.array([[[1, 0], [0, 1]]])):
        self.name = name
        self.state = state
        # base prob for transitions from primary tumour to the node
        self.p = p
        # evolution prob for transitions from microscopic state to macroscopic
        self.ep = ep

        self.narity = obs_table.shape[::-1][0]
        self.n_obs = obs_table.shape[0]
        self.obs_table = obs_table

        self.inc = []
        self.out = []



    def report(self):
        """Just quickly prints infos about the node"""
        print("name: {}, p = {}".format(self.name, self.p))
        print("incoming: ", end="")
        for i in self.inc:
            print("{}, ".format(i.start.name), end="")
        print("\noutgoing: ", end="")
        for o in self.out:
            print("{}, ".format(o.end.name))



    def trans_prob(self, log=False):
        """Computes the transition probabilities from the current state to all
        other possible states.
        
        Parameters
        ----------
        log : bool, default=False
            If True method returns the log-probability
        
        Returns
        -------
        res : numpy array
            Transition probabilities from current state to all three other states
        """
        if self.narity == 2:
            res = np.array([1-self.p, self.p])

            if self.state == 0:
                pass
            else:
                if log:
                    return np.array([-np.inf, 0.])
                else:
                    return np.array([0., 1.])

            for edge in self.inc:
                res[1] += res[0] * edge.t[edge.start.state]

                res[0] *= 1 - edge.t[edge.start.state]

        # new approach
        if self.narity == 3:
            res = np.array([1-self.p, self.p, 0.])

            if self.state == 0:
                for edge in self.inc:
                    res[1] += res[0] * edge.t[edge.start.state]

                    res[0] *= 1 - edge.t[edge.start.state]
            elif self.state == 1:
                if log:
                    return np.array([-np.inf, np.log(1-self.ep), np.log(self.ep)])
                else:
                    return np.array([0., 1-self.ep, self.ep])
            elif self.state == 2:
                if log:
                    return np.array([-np.inf, -np.inf, 0.])
                else:
                    return np.array([0., 0., 1.])

        if log:
            return np.log(res)
        else:
            return res



    def obs_prob(self, log=False, observation=None):
        """Computes the probability of observing a certain modality, given the 
        current state.

        Parameters
        ----------
        log : bool, default=False
            If True, method returns the log-prob.

        observation : array, shape=(n_obs,)
            Contains an observation for each modality.

        Returns
        -------
        res : numpy array
            Observation probability
        """
        res = 1
        if observation is not None:
            for i in range(self.n_obs):
                res *= self.obs_table[i, observation[i], self.state]

        if log:
            return np.log(res)
        else:
            return res



    def bn_prob(self, log=False):
        """Computes the conditional probability of a node being in the state 
        it is in, given its parents are in the states they are in.

        Parameters
        ----------
        log : bool, default=False
            If True, returns the log-probability

        Returns
        -------
        res : float
            Conditional (log-)probability
        """
        res = 1.
        for edge in self.inc:
            res *= (1 - edge.t[edge.start.state])**edge.start.state

        res *= (1 - self.p)*(-1)**self.state
        res += self.state

        if log:
            return np.log(res)
        else:
            return res





class Edge:
    """Class for the connections between lymph node levels (LNLs) represented
    by the Node class.

    Attributes
    ----------
    start : Node instance
        Parent node

    end : Node instance
        Child node

    t_mu : float
        Transition probability in case start-Node has state 1 (microscopic)

    t_M : float
        Transition probability in case start-Node has state 2 (MACROSCOPIC)

    """
    def __init__(self, start, end, t=None, narity=3):
        if type(start) is not Node:
            raise Exception("Start must be Node!")
        if type(end) is not Node:
            raise Exception("End must be Node!")

        self.start = start 
        self.start.out.append(self)
        self.end = end 
        self.end.inc.append(self)

        if t is None:
            self.t = np.zeros(shape=(narity,))
        else:
            self.t = t
            self.t[0] = 0
        for i in range(1, len(self.t)):
            if self.t[i] < self.t[i-1]:
                raise Exception("t cannot decrease!")



    def report(self):
        """Just quickly prints infos about the edge"""
        print("start: {}".format(self.start.name))
        print("end: {}".format(self.end.name))
        print("t = {}".format(self.t))





class System:
    """Class that describes a whole lymphatic system with its lymph node levels 
    (LNLs) and the connections between them.

    Attributes
    ----------
    narity : int
        Node instances (LNLs) can have states in {0, ..., narity-1}

    n_obs : int
        Number of different observation modalities (pathology, CT, MRI, ...)

    nodes : list of Node instances
        Nodes (LNLs) the system is made up of

    edges : list of Edge instances
        Edges that connect the Nodes (LNLs) of the system

    state_list : numpy array, shape=(len(nodes)**narity, len(nodes))
        list of all possible states (more precisely their numerical 
        representation) the system can be in.

    A : numpy array, shape=(len(state_list), len(state_list)), dtype=int
        Square transition matrix with the probabilities to transition from any
        state into any other state.

    idx_dict : dictionary
        for every row of matrix A it gives the index of entries that are NOT zero

    B : numpy array, shape=(len(state_list), len(obs_list))
        Observation matrix with the probabilities to observe any observation
        given any state

    obs_list : numpy array, dtype=int
        Like state_list, but for all possible observations

    C : numpy arrray, shape=(len(obs_list), # of unique diagnoses)
        Matrix for marginalization. If (unique) diagnoses contain unobserved
        LNLs, then one needs to marginalize over all mattching observations. 
        This matrix does so. Each column represents a unique diagnose and each
        row an observation. Each entry can be either 0 or 1, so the computation
        z @ C, where z is a vector of probabilities for each possible obser-
        vation, yields the probability of each unique diagnose in the dataset
        (only for the Bayesian network).

    f : numpy array, shape=(# of unique diagnoses)
        Counts of marginalzied unique observations corresponding to C 
        (only for the Bayesian network).

    C_dict :
        Same as C, but with one C-matrix for every T-stage (only for the 
        hidden Markov model).

    f_dict :
        same as f, but with one f-vector for every T-stage (only for the hidden
        Markov model).   
    """
    def __init__(self, graph={}, obs_table=np.array([[[1. , 0. , 0. ], 
                                                      [0. , 1. , 1. ]], 
                                                     [[0.8, 0.2, 0.2], 
                                                      [0.2, 0.8, 0.8]]])):
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
        """Finds a node with name "name"
        """
        for node in self.nodes:
            if node.name == name:
                return node



    def find_edge(self, startname, endname):
        """finds and edge that starts and end with specific nodes
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
        """Sets the state of the system to newstate.
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

        Parameters
        ---------
        theta : numpy array
            the new parameters that should be fed into the system. The first 
            2*N numbers in the array contain the base probabilities, and evo-
            lution probabilities (if narity = 3), where N is the number of 
            nodes. The remaining entries contain either one or two parameters 
            for each edge in the system (depending on the narity).

        mode : str, default="HMM"
            If one is in "BN" mode (Bayesian network), then it is not necessary
            to compute the transition matrix A again, so it is skipped.
        """
        for i, node in enumerate(self.nodes):
            node.p = theta[(self.narity-1)*i]
            node.ep = theta[(self.narity-1)*i+1]
        for i, edge in enumerate(self.edges):
            edge.t[0] = 0.0
            for j in range(1, self.narity):
                edge.t[j] = theta[(self.narity-1)*len(self.nodes) + i*(self.narity-1) + (j-1)]

        if mode=="HMM":
            self.gen_A()



    def trans_prob(self, newstate, log=False, acquire=False):
        """Computes the probability to transition to newstate, given its 
        current state.

        Parameters
        ----------
        newstate : list
            List of new states for each node in the lymphatic system. The trans-
            ition probability will be computed from the current states to these
            states.

        log : bool, default=False
            if True, the log-probability is computed

        acquire : bool, default=False
            if True, after computing and returning the probability, the system 
            updates its own state to be newstate.

        Returns
        -------
        res : float
            transition probability
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

        Parameters
        ----------
        observation : numpy array, shape=(n_obs, len(self.nodes))
            Contains the observed state of the patient. if there are N 
            different diagnosing modalities and M nodes this has the shape 
            (N, M) and contains zeros and ones.

        log : bool, default=False
            If True, this returns the log probability.

        Returns
        -------
        res : float
            probability to see this observaation
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
        """Generates a dictionary that contains for each row of A those indices 
        where A is NOT zero.
        """
        self.idx_dict = {}
        for i in range(self.narity**len(self.nodes)):
            self.idx_dict[i] = []
            for j in range(self.narity**len(self.nodes)):
                if not np.any(np.greater(self.state_list[i,:], 
                                         self.state_list[j,:])):
                    self.idx_dict[i].append(j)



    def gen_A(self):
        """Generates the transition matrix A, which contains the P(X|X). A is 
        a square matrix with size (# of states). the lower diagonal is zero.
        """
        self.A = np.zeros(shape=(self.narity**len(self.nodes), self.narity**len(self.nodes)))
        for i in range(self.narity**len(self.nodes)):
            self.set_state(self.state_list[i,:])
            for j in self.idx_dict[i]:
                self.A[i,j] = self.trans_prob(self.state_list[j,:])



    def gen_B(self):
        """Generates the observation matrix B, which contains the P(Z|X). B has 
        the shape (# of states, # of possible observations)
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

        Parameters
        ----------
        t_stage : list, default=[1,2,3,4]
            List of T-stages that should be included in the learning process.

        time_dist_dict :  dictionary
            Dictionary with keys of T-stages in t_stage and values of time
            priors for each of those T-stages.

        T : float, default=1
            Temperature of the epoch. Can be reduced from a starting value down
            to almost zero for an annealing approach to sampling.

        Returns
        -------
        likely : float
            log-likelihood of the epoch
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
        p = start @ A^t @ B @ C results in an array of probabilities that can 
        - together with the frequencies f - be used to compute the likelihood. 
        This also works for the Bayesian network case: p = a @ C where a is an 
        array containing the probability for each state.

        Parameters
        ----------
        data : pandas dataframe
            Contains rows of patient data. The columns must include the T-stage 
            and at least one diagnostic modality.

        t_stage : list, default=[1,2,3,4]
            List of T-stages that should be included in the learning process.

        observations : list of str, default=['pCT', 'path']
            List of observational modalities from the pandas dataframe that 
            should be included.

        mode : str, default="HMM"
            "HMM" for hidden Markov model and "BN" for Bayesian network
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

        Parameters
        ----------
        theta : numpy array
            Set of parameters, consisting of the base probabilities (as many as 
            the system has nodes), the transition probabilities (as many as the 
            system has edges) and - not yet implemented - the evolution 
            probabilities (as many as the system has nodes).

        t_stage : list, default=[1,2,3,4]
            List of T-stages that should be included in the learning process.

        time_dist_dict :  dictionary
            Dictionary with keys of T-stages in t_stage and values of time
            priors for each of those T-stages.

        mode : str, default="HMM"
            "HMM" for hidden Markov model and "BN" for Bayesian network

        Returns
        -------
        res : float
            the log-likelihood
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

        Parameters
        ----------
        inv : np.array, shape=(len(self.state_list)) 
            Contains None for node levels that one is not interested in or one 
            of the possible states for the respective involvement.

        obs : np.array, shape=(len(self.obs_list))
            Contains None for node levels where no observation is available and 
            0 or 1 for the respective observation.

        mode : str, default="HMM"
            "HMM" for hidden Markov model and "BN" for Bayesian network

        Returns
        -------
        risk : float
            The risk for the involvement of interest, given an observation
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