import numpy as np
from pandas.core import base
import scipy as sp 
import scipy.stats
from scipy.special import factorial as fact
import pandas as pd
import warnings
from typing import Union, Optional, List, Dict, Any

from .node import Node
from .edge import Edge
from .unilateral import Unilateral


def fast_binomial_pmf(k, n, p):
    """
    Compute the probability mass function of the binomial distribution.
    """
    q = (1 - p)
    binom_coeff = fact(n) / (fact(k) * fact(n - k))
    return binom_coeff * p**k * q**(n - k)
    


# I chose not to make this one a child of System, since it is basically only a 
# container for two System instances
class Bilateral(object):
    """Class that models metastatic progression in a lymphatic system 
    bilaterally by creating two :class:`Unilateral` instances that are 
    symmetric in their connections. The parameters describing the spread 
    probabilities however need not be symmetric.
            
    See Also:
        :class:`Unilateral`: Two instances of this class are created as 
        attributes.
    """
    def __init__(self, 
                 graph: dict = {},
                 base_symmetric: bool = False,
                 trans_symmetric: bool = True):
        """Initialize both sides of the network as a :class:`Unilateral` 
        instance:
        
        Args:
            graph: Dictionary of the same kind as for initialization of 
                :class:`Unilateral`. This graph will be passed to the 
                constructors of two :class:`Unilateral` attributes of this 
                class.
            base_symmetric: If ``True``, the spread probabilities of the two 
                sides from the tumor(s) to the LNLs will be set symmetrically.
            trans_symmetric: If ``True``, the spread probabilities among the 
                LNLs will be set symmetrically.
        """
        self.ipsi   = Unilateral(graph=graph)   # ipsilateral and...
        self.contra = Unilateral(graph=graph)   # ...contralateral network
        
        self.base_symmetric  = base_symmetric
        self.trans_symmetric = trans_symmetric
    
    
    def __str__(self):
        """Print info about the structure and parameters of the bilateral 
        lymphatic system.
        """
        num_tumors = len(self.ipsi.tumors)
        num_lnls   = len(self.ipsi.lnls)
        string = (
            f"Bilateral lymphatic system with {num_tumors} tumor(s) "
            f"and 2 * {num_lnls} LNL(s).\n"
        )
        string += "Symmetry: " 
        string += "base " if self.base_symmetric else ""
        string += "trans\n" if self.trans_symmetric else "\n"
        string += "Ipsilateral:\t" + " ".join([f"{e}" for e in self.ipsi.edges])
        string += "\n"
        string += "Contralateral:\t" + " ".join([f"{e}" for e in self.contra.edges])
        
        return string
    
    
    @property
    def system(self):
        """Return a dictionary with the ipsi- & contralateral side's 
        :class:`Unilateral` under the keys ``"ipsi"`` and ``"contra"`` 
        respectively.
        
        This is needed since in some weak moment, I thought it would be a great 
        idea if a class named ``BilateralSystem`` had an attriute called 
        ``system`` which contained two instances of the ``System`` class under 
        the keys ``"ipsi"`` and ``"contra"``...
        """
        return {
            "ipsi"  : self.ipsi,
            "contra": self.contra
        }
    
    
    @property
    def state(self) -> np.ndarray:
        """
        Return the currently state (healthy or involved) of all LNLs in the 
        system.
        """
        ipsi_state = self.ipsi.state
        contra_state = self.contra.state
        return np.concatenate([ipsi_state, contra_state])
    
    
    @state.setter
    def state(self, newstate: np.ndarray):
        """
        Set the state of the system to ``newstate``.
        """
        self.ipsi.state = newstate[:len(self.ipsi.lnls)]
        self.contra.state = newstate[len(self.ipsi.lnls):]
    
    
    @property
    def base_probs(self) -> np.ndarray:
        """Probabilities of lymphatic spread from the tumor(s) to the lymph 
        node levels. If the ipsi- & contralateral spread from the tumor is set 
        to be symmetric (``base_symmetric = True``) this only returns the 
        parameters of one side. So, the returned array is composed like so:
        
        +-----------------+--------------------+
        | base probs ipsi | base probs contra* |
        +-----------------+--------------------+
        
        *Only when ``base_symmetric = False``, which is the default.
        
        When setting these parameters, the length of the provided array only 
        needs to be half as long if ``base_symmetric`` is ``True``, since both 
        sides will be set to the same values.
        
        See Also:
            :attr:`Unilateral.base_probs`
        """
        if self.base_symmetric:
            return self.ipsi.base_probs
        else:
            return np.concatenate([self.ipsi.base_probs, 
                                   self.contra.base_probs])
    
    @base_probs.setter
    def base_probs(self, new_base_probs: np.ndarray):
        """Set the base probabilities from the tumor(s) to the LNLs.
        """
        if self.base_symmetric:
            self.ipsi.base_probs = new_base_probs
            self.contra.base_probs = new_base_probs
        else:
            num_base_probs = len(self.ipsi.base_edges)
            self.ipsi.base_probs = new_base_probs[:num_base_probs]
            self.contra.base_probs = new_base_probs[num_base_probs:]

    
    @property
    def trans_probs(self) -> np.ndarray:
        """Probabilities of lymphatic spread among the lymph node levels. If 
        this ipsi- & contralateral spread is set to be symmetric 
        (``trans_symmetric = True``) this only returns the parameters of one 
        side. Similiar to the :attr:`base_probs`, this array's shape is:
        
        +------------------+---------------------+
        | trans probs ipsi | trans probs contra* |
        +------------------+---------------------+
        
        *Only if ``trans_symmetric = False``.
        
        And correspondingly, if setting these transmission probability one only 
        needs half as large an array if ``trans_symmetric`` is ``True``.
        
        See Also:
            :attr:`Unilateral.trans_probs`
        """
        if self.trans_symmetric:
            return self.ipsi.trans_probs
        else:
            return np.concatenate([self.ipsi.trans_probs, 
                                   self.contra.trans_probs])
    
    @trans_probs.setter
    def trans_probs(self, new_trans_probs: np.ndarray):
        """Set the transmission probabilities (from LNL to LNL) of the network.
        """
        if self.trans_symmetric:
            self.ipsi.trans_probs = new_trans_probs
            self.contra.trans_probs = new_trans_probs
        else:
            num_trans_probs = len(self.ipsi.trans_edges)
            self.ipsi.trans_probs = new_trans_probs[:num_trans_probs]
            self.contra.trans_probs = new_trans_probs[num_trans_probs:]
    
    
    @property
    def spread_probs(self) -> np.ndarray:
        """The parameters representing the probabilities for lymphatic spread 
        along a directed edge of the graph representing the lymphatic network.
        
        If the bilateral network is set to have symmetries, the length of the 
        list/array of numbers that need to be provided will be shorter. E.g., 
        when the bilateral lymphatic network is completely asymmetric, it 
        requires an array of length :math:`2n_b + 2n_t` where :math:`n_b` is 
        the number of edges from the tumor to the LNLs and :math:`n_t` the 
        number of edges among the LNLs.
        
        Similar to the :attr:`base_probs` and the :attr:`trans_probs`, we can 
        describe its shape like this:
        
        +-----------------+--------------------+------------------+----------------------+
        | base probs ipsi | base probs contra* | trans probs ipsi | trans probs contra** |
        +-----------------+--------------------+------------------+----------------------+
        
        | *Only if ``base_symmetric = False``, which is the default.
        | **Only if ``trans_symmetric = False``.
        
        See Also:
            :attr:`Unilateral.spread_probs`
        """
        return np.concatenate([self.base_probs, self.trans_probs])
    

    @spread_probs.setter
    def spread_probs(self, new_spread_probs: np.ndarray):
        """Set the spread probabilities of the :class:`Edge` instances in the 
        the network.
        """
        num_base_probs = len(self.ipsi.base_edges)
        
        if self.base_symmetric:
            self.base_probs = new_spread_probs[:num_base_probs]
            self.trans_probs = new_spread_probs[num_base_probs:]
        else:
            self.base_probs = new_spread_probs[:2*num_base_probs]
            self.trans_probs = new_spread_probs[2*num_base_probs:]


    @property
    def modalities(self):
        """Compute the two system's observation matrices 
        :math:`\\mathbf{B}^{\\text{i}}` and :math:`\\mathbf{B}^{\\text{c}}`.
                
        See Also:
            :meth:`Unilateral.modalities`: Setting modalities in unilateral 
            System.
        """
        ipsi_modality_spsn = self.ipsi.modalities
        if ipsi_modality_spsn != self.contra.modalities:
            msg = ("Ipsi- & contralaterally stored modalities are not the same")
            raise RuntimeError(msg)
        
        return ipsi_modality_spsn
    
    
    @modalities.setter
    def modalities(self, modality_spsn: Dict[str, List[float]]):
        """
        Given specificity :math:`s_P` & sensitivity :math:`s_N` of different 
        diagnostic modalities, compute the system's two observation matrices 
        :math:`\\mathbf{B}_i` and :math:`\\mathbf{B}_c`.
        """
        self.ipsi.modalities = modality_spsn
        self.contra.modalities = modality_spsn
    
    
    def load_data(
        self,
        data: pd.DataFrame, 
        t_stages: Optional[List[int]] = None, 
        modality_spsn: Optional[Dict[str, List[float]]] = None, 
        mode: str = "HMM"
    ):
        """Load a dataset by converting it into internal representation as data 
        matrix.
        
        Args:
            data: Table with rows of patients. Columns must have three levels. 
                The first column is ('info', 'tumor', 't_stage'). The rest of 
                the columns are separated by modality names on the top level, 
                then subdivided into 'ipsi' & 'contra' by the second level and
                finally, in the third level, the names of the lymph node level 
                are given. Here is an example of such a table: 
                
                +---------+---------------------+-----------------------+
                |  info   |         MRI         |         PET           |
                +---------+----------+----------+-----------+-----------+
                |  tumor  |   ipsi   |  contra  |   ipsi    |  contra   |
                +---------+----------+----------+-----------+-----------+
                | t_stage |    II    |    II    |    II     |    II     |
                +=========+==========+==========+===========+===========+
                | early   | ``True`` | ``None`` | ``True``  | ``False`` |
                +---------+----------+----------+-----------+-----------+
                | late    | ``None`` | ``None`` | ``False`` | ``False`` |
                +---------+----------+----------+-----------+-----------+
                | early   | ``True`` | ``True`` | ``True``  | ``None``  |
                +---------+----------+----------+-----------+-----------+
        
        See Also:
            :meth:`Unilateral.load_data`: Data loading method of unilateral 
            system.
            
            :meth:`Unilateral._gen_C`: Generate marginalization matrix.
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
        self.ipsi.load_data(
            ipsi_data, 
            t_stages=t_stages,
            modality_spsn=modality_spsn,
            mode=mode,
            gen_C_kwargs=gen_C_kwargs
        )
        self.contra.load_data(
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
    

    def _log_likelihood(
        self,
        t_stages: Optional[List[Any]] = None,
        diag_times: Optional[Dict[Any, int]] = None,
        max_t: Optional[int] = 10,
        time_dists: Optional[Dict[Any, np.ndarray]] = None
    ):
        """Compute the log-likelihood of data, using the stored spread probs. 
        This method mainly exists so that the checking and assigning of the 
        spread probs can be skipped.
        """
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
                state_probs["ipsi"] = self.ipsi._evolve(diag_time)
                state_probs["contra"] = self.contra._evolve(diag_time)
                
                # matrix with joint probabilities for any combination of ipsi- 
                # & conralateral diagnoses after given number of time-steps
                joint_diagnose_prob = (
                    self.ipsi.B.T 
                    @ np.outer(state_probs["ipsi"], state_probs["contra"])
                    @ self.contra.B
                )
                log_p = np.log(
                    np.sum(
                        self.ipsi.C[stage]
                        * (joint_diagnose_prob 
                           @ self.contra.C[stage]),
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
            state_probs["ipsi"] = self.ipsi._evolve(t_last=max_t)
            state_probs["contra"] = self.contra._evolve(t_last=max_t)
            
            for stage in t_stages:
                joint_diagnose_prob = (
                    self.ipsi.B.T
                    @ state_probs["ipsi"].T
                    @ np.diag(time_dists[stage])
                    @ state_probs["contra"]
                    @ self.contra.B
                )
                log_p = np.log(
                    np.sum(
                        self.ipsi.C[stage]
                        * (joint_diagnose_prob 
                           @ self.contra.C[stage]),
                        axis=0
                    )
                )
                llh += np.sum(log_p)
            
            return llh
        
        else:
            msg = ("Either provide a list of diagnose times for each T-stage "
                   "or a distribution over diagnose times for each T-stage.")
            raise ValueError(msg)
    
    
    def log_likelihood(
        self,
        spread_probs: np.ndarray,
        t_stages: Optional[List[Any]] = None,
        diag_times: Optional[Dict[Any, int]] = None,
        max_t: Optional[int] = 10,
        time_dists: Optional[Dict[Any, np.ndarray]] = None
    ):
        """Compute log-likelihood of (already stored) data, given the spread 
        probabilities and either a discrete diagnose time or a distribution to 
        use for marginalization over diagnose times.
        
        Args:
            spread_probs: Spread probabiltites from the tumor to the LNLs, as 
                well as from (already involved) LNLs to downsream LNLs. Includes 
                both sides of the neck. The composition of this array is:
                
                +----------------------------+-----------------------------+
                | base probs (ipsi & contra) | trans probs (ipsi & contra) |
                +----------------------------+-----------------------------+
                
                If certain symmetries are chosen, only one set of base or 
                transmission probabilities might have to be provided.
            
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
        
        Returns:
            The log-likelihood :math:`\\log{p(D \\mid \\theta)}` where :math:`D` 
            is the data and :math:`\\theta` is the tuple of spread probabilities 
            and diagnose times or distributions over diagnose times.
        
        See Also:
            :attr:`spread_probs`: Property for getting and setting the spread 
            probabilities, of which a lymphatic network has as many as it has 
            :class:`Edge` instances (in case no symmetries apply).
            
            :meth:`Unilateral.log_likelihood`: The log-likelihood function of 
            the unilateral system.
        """
        if not self._spread_probs_are_valid(spread_probs):
            return -np.inf
        
        self.spread_probs = spread_probs
        
        if t_stages is None:
            t_stages = list(self.ipsi.f.keys())
        
        return self._log_likelihood(
            t_stages=t_stages,
            diag_times=diag_times,
            max_t=max_t,
            time_dists=time_dists,
        )
            

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
            theta: Spread probabiltites from the tumor to the LNLs, as well as 
                from (already involved) LNLs to downsream LNLs. Includes both 
                sides of the neck. The composition of this array is:
                
                +----------------------------+-----------------------------+
                | base probs (ipsi & contra) | trans probs (ipsi & contra) |
                +----------------------------+-----------------------------+
                
                If certain symmetries are chosen, only one set of base or 
                transmission probabilities might have to be provided.

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
                and the time of diagnosis for all T-stages. It is therefore 
                made up these parts and in that order:
                
                +------------+-------------+----------------+
                | base probs | trans probs | diagnose times |
                +------------+-------------+----------------+
                
            t_stages: keywords of T-stages that are present in the dictionary of 
                C matrices and the previously loaded dataset.
            
            max_t: Latest accepted time-point.
            
        Returns:
            The likelihood of the data, given the spread parameters as well as 
            the diagnose time for each T-stage.
        
        See Also:
            :meth:`log_likelihood`: The `theta` argument of this function is 
            split into `spread_probs` and `diag_times`, which are then passed 
            to the actual likelihood function.
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
                and the binomial distribution's :math:`p` parameters for each 
                T-category. One has to provide a concatenated array of these 
                numbers like this:
                
                +------------+-------------+--------------------------+
                | base probs | trans probs | binomial :math`p` params |
                +------------+-------------+--------------------------+
                
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
               @ self.ipsi.B.T
               @ pXX
               @ self.contra.B
               @ cZ["contra"])
        
        return pDDII / pDD



class BilateralSystem(Bilateral):
    """Class kept for compatibility after renaming to :class:`Bilateral`.
    
    See Also:
        :class:`Bilateral`
    """
    def __init__(self, *args, **kwargs):
        msg = ("This class has been renamed to `Bilateral`.")
        warnings.warn(msg, DeprecationWarning)
        
        super().__init__(*args, **kwargs)