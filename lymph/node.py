import numpy as np
import scipy as sp 
import scipy.stats

class Node(object):
    """Class for lymph node levels (LNLs) in a lymphatic system.

    Args:
        name (str): Name of the node

        state (int): Current state this LNL is in. Can be in {0, ..., narity-1}

        p (float): Base probability :math:`b`. Probability that this LNL gets 
            infected from the primary tumour.

        ep (float): Evolution probability :math:`e`. In case of narity = 3 this 
            is the probability that the LNL spontaneously develops from state = 1 
            into state = 2.

        obs_table (numpy array, 3D): A 2D arrray for each observational modality.
            These 2D arrays contain the conditional probabilities of observations 
            given hidden states (like sensitivity :math:`s_N` and specificity 
            :math:`s_P`). Each of the 2D arrays must be "column-stochastic".
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
        
        Args:
            log (bool): If ``True`` method returns the log-probability.
                (default: ``False``)
        
        This method returns the transition probabilities from current state to 
        all two (three) other states.
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

        Args:
            log (bool): If ``True``, method returns the log-prob.

            observation (array): Contains an observation for 
                each modality. ``shape=(n_obs,)``
                (default: ``False``)

        This method returns the observation probability.
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

        Args:
            log (bool): If ``True``, returns the log-probability.
                (default: ``False``)

        This method returns the conditional (log-)probability.
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