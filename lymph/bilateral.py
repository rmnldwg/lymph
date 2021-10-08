import numpy as np
import scipy as sp 
import scipy.stats
from scipy.special import factorial as fact
import pandas as pd
import warnings
from typing import Union, Optional, List, Dict, Any

from .node import Node
from .edge import Edge
from .unilateral import System


def fast_binomial_pmf(k, n, p):
    """
    Compute the probability mass function of the binomial distribution.
    """
    q = (1 - p)
    binom_coeff = fact(n) / (fact(k) * fact(n - k))
    return binom_coeff * p**k * q**(n - k)
    


# I chose not to make this one a child of System, since it is basically only a 
# container for two System instances
class BilateralSystem(object):
    """Class that models metastatic progression in a lymphatic system 
    bilaterally by creating two :class:`System` instances that are symmetric in 
    their connections. The parameters describing the spread probabilities 
    however need not be symmetric.
            
    See Also:
        :class:`System`: Two instances of this class are created as attributes.
    """
    def __init__(self, 
                 graph: dict = {},
                 base_symmetric: bool = False,
                 trans_symmetric: bool = True):
        """Initialize both sides of the network as :class:`System` instances:
        
        Args:
            graph: Dictionary of the same kind as for initialization of 
                :class:`System`. This graph will be passed to the constructors of 
                two :class:`System` attributes of this class.
            base_symmetric: If ``True``, the spread probabilities of the two 
                sides from the tumor(s) to the LNLs will be set symmetrically.
            trans_symmetric: If ``True``, the spread probabilities among the 
                LNLs will be set symmetrically.
        """
        self.system = {}
        self.system["ipsi"] = System(graph=graph)   # ipsilateral and...
        self.system["contra"] = System(graph=graph)   # ...contralateral part of the network
        
        # sort all the edges into the two sides (ipsi & contra, first index) 
        # and into base (from tumor to LNL) & trans (from LNL to LNL) along the 
        # second index. 
        self.edges = np.empty(shape=(2,2), dtype=list)
        
        # ipsi
        ipsi_base, ipsi_trans = [], []
        for e in self.system["ipsi"].edges:
            if e.start.typ == 'tumor':
                ipsi_base.append(e)
            elif e.start.typ == 'lnl':
                ipsi_trans.append(e)
            else:
                raise Exception(f"Node {e.start.name} has no correct type.")
        self.edges[0,0] = ipsi_base
        self.edges[0,1] = ipsi_trans
        
        # contra
        contra_base, contra_trans = [], []
        for e in self.system["contra"].edges:
            if e.start.typ == 'tumor':
                contra_base.append(e)
            elif e.start.typ == 'lnl':
                contra_trans.append(e)
            else:
                raise Exception(f"Node {e.start.name} has no correct type.")
        self.edges[1,0] = contra_base
        self.edges[1,1] = contra_trans
        
        self.base_symmetric = base_symmetric
        self.trans_symmetric = trans_symmetric
    
    
    def __str__(self):
        string = "### IPSILATERAL ###\n"
        string += self.system["ipsi"].__str__()
        string += "\n### CONTRALATERAL ###\n"
        string += self.system["contra"].__str__()
        return string
    
    
    @property
    def state(self) -> np.ndarray:
        """
        Return the currently state (healthy or involved) of all LNLs in the 
        system.
        """
        ipsi_state = self.system["ipsi"].state
        contra_state = self.system["contra"].state
        return np.concatenate([ipsi_state, contra_state])
    
    
    @state.setter
    def state(self, newstate: np.ndarray):
        """
        Set the state of the system to ``newstate``.
        """
        self.system["ipsi"].state = newstate[:len(self.system["ipsi"].lnls)]
        self.system["contra"].state = newstate[len(self.system["ipsi"].lnls):]
    
    
    @property
    def spread_probs(self) -> np.ndarray:
        """
        Return the spread probabilities of the :class:`Edge` instances in 
        the network. Length and structure of ``theta`` depends on the set 
        symmetries of the network.
        
        See Also:
            :meth:`theta`: Setting the spread probabilities and symmetries.
        """
        spread_probs = np.array([], dtype=float)
        switch = [self.base_symmetric, self.trans_symmetric]
        for edge_type in [0,1]:
            tmp = [e.t for e in self.edges[0,edge_type]]
            spread_probs = np.append(spread_probs, tmp.copy())
            if not switch[edge_type]:
                tmp = [e.t for e in self.edges[1,edge_type]]
                spread_probs = np.append(spread_probs, tmp.copy())
        
        return spread_probs
    

    @spread_probs.setter
    def spread_probs(self, spread_probs: np.ndarray):
        """
        Set the spread probabilities of the :class:`Edge` instances in the 
        the network.
        """
        switch = [self.base_symmetric, self.trans_symmetric]
        cursor = 0
        for edge_type in [0,1]:
            for side in [0,1]:
                num = len(self.edges[side,edge_type])
                tmp = spread_probs[cursor:cursor+num]
                for i,e in enumerate(self.edges[side,edge_type]):
                    e.t = tmp[i]
                if switch[edge_type]:
                    for i,e in enumerate(self.edges[1,edge_type]):
                        e.t = tmp[i]
                    cursor += num
                    break
                else:
                    cursor += num
        
        for side in ["ipsi", "contra"]:
            try:
                self.system[side]._gen_A()
            except AttributeError:
                n = len(self.system[side].state_list)
                self.system[side].A = np.zeros(shape=(n,n))
                self.system[side]._gen_A()


    @property
    def modalities(self):
        """
        Compute the two system's observation matrices 
        :math:`\\mathbf{B}^{\\text{i}}` and :math:`\\mathbf{B}^{\\text{c}}`.
                
        See Also:
            :meth:`System.set_modalities`: Setting modalities in unilateral System.
        """
        ipsi_modality_spsn = self.system["ipsi"].modalities
        if ipsi_modality_spsn != self.system["contra"].modalities:
            msg = ("Ipsi- & contralaterally stored modalities are not the same")
            raise RuntimeError(msg)
        
        return ipsi_modality_spsn
    
    
    @modalities.setter
    def modalities(self, modality_spsn: Dict[Any, List[float]]):
        """
        Given specificity :math:`s_P` & sensitivity :math:`s_N` of different 
        diagnostic modalities, compute the system's two observation matrices 
        :math:`\\mathbf{B}_i` and :math:`\\mathbf{B}_c`.
        """
        self.system["ipsi"].modalities = modality_spsn
        self.system["contra"].modalities = modality_spsn
    
    
    def load_data(
        self,
        data: pd.DataFrame, 
        t_stages: Optional[List[int]] = None, 
        modality_spsn: Optional[Dict[str, List[float]]] = None, 
        mode: str = "HMM"
    ):
        """
        Args:
            data: Table with rows of patients. Columns must have three levels. 
                The first column is ('Info', 'tumor', 'T-stage'). The rest of 
                the columns are separated by modality names on the top level, 
                then subdivided into 'ipsi' & 'contra' by the second level and
                finally, in the third level, the names of the lymph node level 
                are given.
        
        See Also:
            :meth:`System.load_data`: Data loading method of unilateral system.
            
            :meth:`System._gen_C`: Generate marginalization matrix.
        """
        # split the DataFrame into two, one for ipsi-, one for contralateral
        ipsi_data = data.drop(
            columns=["contra"], axis=1, level=1, inplace=False
        )
        ipsi_data = pd.DataFrame(
            ipsi_data.values,
            index=ipsi_data.index, 
            columns=ipsi_data.columns.droplevel(1)
        )
        contra_data = data.drop(
            columns=["ipsi"], axis=1, level=1, inplace=False
        )
        contra_data = pd.DataFrame(
            contra_data.values,
            index=contra_data.index, 
            columns=contra_data.columns.droplevel(1)
        )
        
        # generate both side's C matrix with duplicates and ones
        gen_C_kwargs = {'delete_ones': False, 'aggregate_duplicates': False}
        self.system["ipsi"].load_data(
            ipsi_data, 
            t_stages=t_stages,
            modality_spsn=modality_spsn,
            mode=mode,
            gen_C_kwargs=gen_C_kwargs
        )
        self.system["contra"].load_data(
            contra_data, 
            t_stages=t_stages, 
            modality_spsn=modality_spsn,
            mode=mode,
            gen_C_kwargs=gen_C_kwargs
        )
    
    
    def _spread_probs_are_valid(self, new_spread_probs: np.ndarray) -> bool:
        """Check that the spread probability (rates) are all within limits.
        """
        if new_spread_probs.shape != self.spread_probs.shape:
            msg = ("Shape of provided spread parameters does not match network")
            raise ValueError(msg)
        if np.any(np.greater(0., new_spread_probs)):
            return False
        if np.any(np.greater(new_spread_probs, 1.)):
            return False
        
        return True
    
    
    def log_likelihood(
        self,
        spread_probs: np.ndarray,
        t_stages: Optional[List[Any]] = None,
        diag_times: Optional[Dict[Any, int]] = None,
        max_t: Optional[int] = 10,
        time_dists: Optional[Dict[Any, np.ndarray]] = None,
        model: str = "HMM"
    ):
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
                on, but can also be more abstract like 'early', 'late' etc.
            
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
        
        See Also:
            :meth:`System.log_likelihood`: The log-likelihood function of the 
                unilateral system.
        """
        if not self._spread_probs_are_valid(spread_probs):
            return -np.inf
        
        if t_stages is None:
            t_stages = list(self.system["ipsi"].f_dict.keys())
        
        self.spread_probs = spread_probs
        
        llh = 0.
        
        if diag_times is not None:
            if len(diag_times) != len(t_stages):
                msg = ("One diagnose time must be provided for each T-stage.")
                raise ValueError(msg)
            
            for stage in t_stages:
                diag_time = np.around(diag_times[stage]).astype(int)
                if diag_time > max_t:
                    return -np.inf
                
                # probabilities for any hidden state (ipsi- & contralaterally)
                state_probs = {}
                state_probs["ipsi"] = self.system["ipsi"]._evolve(diag_time)
                state_probs["contra"] = self.system["contra"]._evolve(diag_time)
                
                # matrix with joint probabilities for any combination of ipsi- 
                # & conralateral diagnoses after given number of time-steps
                joint_diagnose_prob = (
                    self.system["ipsi"].B.T 
                    @ np.outer(state_probs["ipsi"], state_probs["contra"])
                    @ self.system["contra"].B
                )
                log_p = np.log(
                    np.sum(
                        self.system["ipsi"].C_dict[stage]
                        * (joint_diagnose_prob 
                           @ self.system["contra"].C_dict[stage]),
                        axis=0
                    )
                )
                llh += np.sum(log_p)
            
            return llh
        
        elif time_dists is not None:
            if len(time_dists) != len(t_stages):
                msg = ("One distribution over diagnose times must be provided "
                       "for each T-stage.")
                raise ValueError(msg)
            
            # subtract 1, to also consider healthy starting state (t = 0)
            max_t = len(time_dists[t_stages[0]]) - 1
            
            state_probs = {}
            state_probs["ipsi"] = self.system["ipsi"]._evolve(t_last=max_t)
            state_probs["contra"] = self.system["contra"]._evolve(t_last=max_t)
            
            for stage in t_stages:
                joint_diagnose_prob = (
                    self.system["ipsi"].B.T
                    @ state_probs["ipsi"].T
                    @ np.diag(time_dists[stage])
                    @ state_probs["contra"]
                    @ self.system["contra"].B
                )
                log_p = np.log(
                    np.sum(
                        self.system["ipsi"].C_dict[stage]
                        * (joint_diagnose_prob 
                           @ self.system["contra"].C_dict[stage]),
                        axis=0
                    )
                )
                llh += np.sum(log_p)
            
            return llh
        
        else:
            msg = ("Either provide a list of diagnose times for each T-stage "
                   "or a distribution over diagnose times for each T-stage.")
            raise ValueError(msg)
            

    def marginal_log_likelihood(
        self, 
        theta: np.ndarray, 
        t_stages: Optional[List[Any]] = None, 
        time_dists: dict = {}
    ) -> float:
        """
        Compute the likelihood of the (already stored) data, given the spread 
        parameters, marginalized over time of diagnosis via time distributions. 
        Wraps the :meth:`log_likelihood` method.

        Args:
            theta: Set of parameters, consisting of the base probabilities 
                :math:`b` (as many as the system has nodes) and the transition 
                probabilities :math:`t` (as many as the system has edges).

            t_stages: List of T-stages that should be included in the learning 
                process.

            time_dists: Distribution over the probability of diagnosis at 
                different times :math:`t` given T-stage.

        Returns:
            The log-likelihood of a parameter sample.

        See Also:
            :meth:`log_likelihood`: Simply calls the actual likelihood function 
                where it sets the `diag_times` to `None`.
        """
        return self.log_likelihood(
            theta, t_stages,
            diag_times=None, time_dists=time_dists
        )
    
    
    def time_log_likelihood(
        self, 
        theta: np.ndarray, 
        t_stages: List[Any],
        max_t: int = 10
    ) -> float:
        """
        Compute likelihood given the spread parameters and the time of diagnosis 
        for each T-stage. Wraps the :math:`log_likelihood` method.
        
        Args:
            theta: Set of parameters, consisting of the spread probabilities 
                (as many as the system has :class:`Edge` instances) and the 
                time of diagnosis for all T-stages.
                
            t_stages: keywords of T-stages that are present in the dictionary of 
                C matrices and the previously loaded dataset.
            
            max_t: Latest accepted time-point.
            
        Returns:
            The likelihood of the data, given the spread parameters as well as 
            the diagnose time for each T-stage.
        
        See Also:
            :math:`log_likelihood`: The `theta` argument of this function is 
                split into `spread_probs` and `diag_times`, which are then 
                passed to the actual likelihood function.
        """
        # splitting theta into spread parameters and...
        len_spread_probs = len(theta) - len(t_stages)
        spread_probs = theta[:len_spread_probs]
        # ...diagnose times for each T-stage
        tmp = theta[len_spread_probs:]
        diag_times = {t_stages[t]: tmp[t] for t in range(len(t_stages))}
        
        return self.log_likelihood(
            spread_probs, t_stages,
            diag_times=diag_times, max_t=max_t, time_dists=None
        )
    
    
    def binom_marg_log_likelihood(
        self, 
        theta: np.ndarray, 
        t_stages: List[Any],
        max_t: int = 10
    ) -> float:
        """
        Compute marginal log-likelihood using binomial distributions to sum 
        over the diagnose times.
        
        Args:
            theta: Set of parameters, consisting of the spread probabilities 
                (as many as the system has :class:`Edge` instances) and the 
                binomial distribution's :math:`p` parameters.
                
            t_stages: keywords of T-stages that are present in the dictionary of 
                C matrices and the previously loaded dataset.
                
            max_t: Latest accepted time-point.
            
        Returns:
            The log-likelihood of the (already stored) data, given the spread 
            prbabilities as well as the parameters for binomial distribtions 
            used to marginalize over diagnose times.
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
    
    
    def risk(
        self,
        spread_probs: Optional[np.ndarray] = None,
        inv: Dict[str, Optional[np.ndarray]] = {"ipsi": None, "contra": None},
        diagnoses: Dict[str, Dict] = {"ipsi": {}, "contra": {}},
        diag_time: Optional[int] = None,
        time_dist: Optional[np.ndarray] = None,
        mode: str = "HMM"
    ) -> float:
        """Compute risk of ipsi- & contralateral involvement given specific (but 
        potentially incomplete) diagnoses for each side of the neck.
        
        Args:
            spread_probs: Set of new spread parameters. If not given (``None``), 
                the currently set parameters will be used.
                
            inv: Dictionary that can have the keys ``"ipsi"`` and ``"contra"`` 
                with the respective values being the involvements of interest. 
                If (for one side or both) no involvement of interest is given, 
                it'll be marginalized.
                The array themselves may contain ``True``, ``False`` or ``None`` 
                for each LNL corresponding to the risk for involvement, no 
                involvement and "not interested".
                
            diagnoses: Dictionary that itself may contain two dictionaries. One 
                with key "ipsi" and one with key "contra". The respective value 
                is then a dictionary that can hold a potentially incomplete 
                (mask with ``None``) diagnose for every available modality. 
                Leaving out available modalities will assume a completely 
                missing diagnosis.
                
            diag_time: Time of diagnosis. Either this or the `time_dist` to 
                marginalize over diagnose times must be given.
                
            time_dist: Distribution to marginalize over diagnose times. Either 
                this, or the `diag_time` must be given.
                
            mode: Set to ``"HMM"`` for the hidden Markov model risk (requires 
                the ``time_dist``) or to ``"BN"`` for the Bayesian network 
                version.
        """
        if spread_probs is not None:
            self.spread_probs = spread_probs
            
        cX = {}   # marginalize over matching complete involvements.
        cZ = {}   # marginalize over Z for incomplete diagnoses.
        pXt = {}  # probability p(X|t) of state X at time t as 2D matrices
        pD = {}   # probability p(D|X) of a (potentially incomplete) diagnose, 
                  # given an involvement. Should be a 1D vector
                  
        for side in ["ipsi", "contra"]:            
            involvement = np.array(inv[side])
            # build vector to marginalize over involvements
            cX[side] = np.zeros(shape=(len(self.system[side].state_list)),
                                dtype=bool)
            for i,state in enumerate(self.system[side].state_list):
                cX[side][i] = np.all(
                    np.equal(
                        involvement, state,
                        where=(involvement!=None),
                        out=np.ones_like(involvement, dtype=bool)
                    )
                )
                
            # create one large diagnose vector from the individual modalitie's
            # diagnoses
            obs = np.array([])
            for mod in self.system[side]._modality_tables:
                if mod in diagnoses[side]:
                    obs = np.append(obs, diagnoses[side][mod])
                else:
                    obs = np.append(obs, np.array([None] * len(self.system[side].lnls)))
            
            # build vector to marginalize over diagnoses
            cZ[side] = np.zeros(shape=(len(self.system[side].obs_list)), 
                                dtype=bool)
            for i,complete_obs in enumerate(self.system[side].obs_list):
                cZ[side][i] = np.all(
                    np.equal(
                        obs, complete_obs,
                        where=(obs!=None), 
                        out=np.ones_like(obs, dtype=bool)
                    )
                )
            
            if diag_time is not None:
                pXt[side] = self.system[side]._evolve(diag_time)
                
            elif time_dist is not None:
                max_t = len(time_dist)
                pXt[side] = self.system[side]._evolve(t_last=max_t-1)
            
            else:
                msg = ("Either diagnose time or distribution to marginalize "
                       "over it must be given.")
                raise ValueError(msg)
            
            pD[side] = self.system[side].B @ cZ[side]
        
        # joint probability of Xi & Xc (marginalized over time). Acts as prior 
        # for p( Di,Dc | Xi,Xc ) and should be a 2D matrix
        if diag_time is not None:
            pXX = np.outer(pXt["ipsi"], pXt["contra"])
            
        elif time_dist is not None:
            # time-prior in diagnoal matrix form
            PT = np.diag(time_dist)
            pXX = pXt["ipsi"].T @ PT @ pXt["contra"]
        
        # joint probability of all hidden states and the requested diagnosis
        pDDXX = np.einsum("i,ij,j->ij", pD["ipsi"], pXX, pD["contra"])
        # joint probability of the requested involvement and diagnosis
        pDDII = cX["ipsi"].T @ pDDXX @ cX["contra"]
        
        # denominator p(Di, Dc). Joint probability for ipsi- & contralateral 
        # diagnoses. Marginalized over all hidden involvements and over all 
        # matching complete observations that give rise to the specific 
        # diagnose. The result should be just a number
        pDD = (cZ["ipsi"].T
               @ self.system["ipsi"].B.T
               @ pXX
               @ self.system["contra"].B
               @ cZ["contra"])
        
        return pDDII / pDD
