import numpy as np
import scipy as sp 
import scipy.stats
import pandas as pd
import warnings
from typing import Union, Optional, List, Dict

from .node import Node
from .edge import Edge
from .unilateral import System


class BilateralSystem(System):
    """Class that describes a bilateral lymphatic system with its lymph node 
    levels (LNLs) and the connections between them. It inherits most attributes 
    and methods from the normal :class:`System`, but automatically creates the 
    :class:`Node` and :class:`Edge` instances for the contralateral side and 
    also manages the symmetric or asymmetric assignment of probabilities to the 
    directed arcs.

    Args:
        graph: For every key in the dictionary, the :class:`BilateralSystem` 
            will create two :class:`Node` instances that represent binary 
            random variables each. One for the ipsilateral (which gets the 
            suffix ``_i``) and one for the contralateral side (suffix ``_c``) of 
            the patient. The values in the dictionary should then be the a list 
            of :class:`Node` names to which :class:`Edge` instances from the 
            current :class:`Node` should be created. The tumor will of course 
            not be mirrored.
    """
    def __init__(self, 
                 graph: dict = {}):
        
        # creating a graph that copies the ipsilateral entries for the 
        # contralateral side as well
        bilateral_graph = {}
        for key, value in graph.items():
            if key[0] == "tumor":
                bilateral_graph[key] = [f"{lnl}_i" for lnl in value]
                bilateral_graph[key] += [f"{lnl}_c" for lnl in value]
            else:
                ipsi_key = ('lnl', f"{key[1]}_i")
                bilateral_graph[ipsi_key] = [f"{lnl}_i" for lnl in value]
                contra_key = ('lnl', f"{key[1]}_c")
                bilateral_graph[contra_key] = [f"{lnl}_c" for lnl in value]
                
        # call parent's initialization with extended graph
        super().__init__(graph=bilateral_graph)
        
        if len(self.lnls) % 2 != 0:
            raise RuntimeError("Number of LNLs should be divisible by 2, but "
                               "isn't.")
            
        # a priori, assume no cross connections between the two sides. So far, 
        # this should not be the case, as it would require quite some changes 
        # here and there, but it's probably good to have it set up.
        no_cross = True
            
        # sort edges into four sets, based on where they start and on which 
        # side they are:
        self.ipsi_base = []      # connections from tumor to ipsilateral LNLs
        self.ipsi_trans = []     # connections among ipsilateral LNLs
        self.contra_base = []    # connections from tumor to contralateral LNLs
        self.contra_trans = []   # connections among contralateral LNLs
        for edge in self.edges:
            if "_i" in edge.end.name:
                if edge.start.typ == "tumor":
                    self.ipsi_base.append(edge)
                elif edge.start.typ == "lnl":
                    self.ipsi_trans.append(edge)
                else:
                    raise RuntimeError(f"Node {edge.start.name} has no typ "
                                       "assigned")
                    
                # check if there are cross-connections from contra to ipsi
                if "_c" in edge.start.name:
                    no_cross = False
                    
            elif "_c" in edge.end.name:
                if edge.start.typ == "tumor":
                    self.contra_base.append(edge)
                elif edge.start.typ == "lnl":
                    self.contra_trans.append(edge)
                else:
                    raise RuntimeError(f"Node {edge.start.name} has no typ "
                                       "assigned")
                    
                # check if there are cross-connections from ipsi to contra
                if "_i" in edge.start.name:
                    no_cross = False
                    
            else:
                raise RuntimeError(f"LNL {edge.end.name} is not assigned to "
                                   "ipsi- or contralateral side")
                
        # if therea re no cross-connections, it makes sense to save the 
        # original graph
        if no_cross:
            self.unilateral_system = System(graph=graph)
                

        
    def get_theta(self,
                  output: str = "dict",
                  order: str = "by_type") -> Union[dict, list]:
        """Return the transition probabilities of all edges in the graph.
        
        Args:
            output: Can be ``"dict"`` or ``"list"``. If the former, they keys 
                will be of the descriptive format ``{start}->{end}`` and contain 
                the respective value. If it's the latter, they will be put in 
                an array in the following order: First the transition 
                probabilities from tumor to LNLs (and within that, first ipsi- 
                then contralateral) then the probabilities for spread among the 
                LNLs (again first ipsi, then contra). A list or array of that 
                format is expected by the method :method:`set_theta(new_theta)`.
                
            order: Option how to order the blocks of transition probabilities. 
                The default ``"by_type"`` will first return all base 
                probabilities (from tumor to LNL) for both sides and then all 
                transition probabilities (among LNLs). ``"by_side"`` will first 
                return both types of probabilities ipsilaterally and then 
                contralaterally. If output is ``"dict"``, this option has no 
                effect.
        """
        
        if output == "dict":
            theta_dict = {}
            # loop in the correct order through the blocks of probabilities & 
            # add the values to semantic/descriptive keys
            for edge in self.ipsi_base:
                start, end = edge.start.name, edge.end.name
                theta_dict[f"{start}->{end}"] = edge.t
            for edge in self.contra_base:
                start, end = edge.start.name, edge.end.name
                theta_dict[f"{start}->{end}"] = edge.t
            for edge in self.ipsi_trans:
                start, end = edge.start.name, edge.end.name
                theta_dict[f"{start}->{end}"] = edge.t
            for edge in self.contra_trans:
                start, end = edge.start.name, edge.end.name
                theta_dict[f"{start}->{end}"] = edge.t
                
            return theta_dict
        
        else:
            theta_list = []
            # loop in the correct order through the blocks of probabilities & 
            # append them to the list
            for edge in self.ipsi_base:
                theta_list.append(edge.t)
            
            if order == "by_side":
                for edge in self.ipsi_trans:
                    theta_list.append(edge.t)
                for edge in self.contra_base:
                    theta_list.append(edge.t)
            elif order == "by_type":
                for edge in self.contra_base:
                    theta_list.append(edge.t)
                for edge in self.ipsi_trans:
                    theta_list.append(edge.t)
            else:
                raise ValueError("Order option must be \'by_type\' or "
                                 "\'by_side\'.")
                
            for edge in self.contra_trans:
                theta_list.append(edge.t)
                
            return np.array(theta_list)
        
        
        
    def set_theta(self, 
                  theta: np.ndarray, 
                  base_symmetric: bool = False,
                  trans_symmetric: bool = True,
                  mode: str = "HMM"):
        """Fills the system with new base and transition probabilities and also 
        computes the transition matrix A again, if one is in mode "HMM". 
        Parameters for ipsilateral and contralateral side can be chosen to be 
        symmetric or independent.

        Args:
            theta: The new parameters that should be fed into the system. They 
                all represent the transition probabilities along the edges of 
                the network and will be set in the following order: First the 
                connections between tumor & LNLs ipsilaterally, then 
                contralaterally, followed by the connections among the LNLs 
                (again first ipsilaterally and then contraleterally).
                If ``base_symmetric`` and/or ``trans_symmetric`` are set to 
                ``True``, the respective block of parameters will be set to both 
                ipsi- & contralateral connections. The lentgh of ``theta`` 
                should then be shorter accordingly.
                
            base_symmetric: If ``True``, base probabilities will be the same for 
                ipsi- & contralateral.
                
            trans_symmetric: If ``True``, transition probability among LNLs will 
                be the same for ipsi- & contralateral.

            mode: If one is in "BN" mode (Bayesian network), then it is not 
                necessary to compute the transition matrix A again, so it is 
                skipped. (default: ``"HMM"``)
        """
        
        cursor = 0
        if base_symmetric:
            for i in range(len(self.ipsi_base)):
                self.ipsi_base[i].t = theta[cursor]
                self.contra_base[i].t = theta[cursor]
                cursor += 1
        else:
            for edge in self.ipsi_base:
                edge.t = theta[cursor]
                cursor += 1
            for edge in self.contra_base:
                edge.t = theta[cursor]
                cursor += 1
                
        if trans_symmetric:
            for i in range(len(self.ipsi_trans)):
                self.ipsi_trans[i].t = theta[cursor]
                self.contra_trans[i].t = theta[cursor]
                cursor += 1
        else:
            for edge in self.ipsi_trans:
                edge.t = theta[cursor]
                cursor += 1
            for edge in self.contra_trans:
                edge.t = theta[cursor]
                cursor += 1
                
        if mode == "HMM":
            self._gen_A()
            
            
    
    def _gen_A(self):
        """Generates the transition matrix of the bilateral lymph system. If 
        ipsi- & contralateral side have no cross connections, this amounts to 
        computing two transition matrices, one for each side. Otherwise, this 
        function will call its parent's method.
        """
        
        # this class should only have a unilateral_system as an attribute, if 
        # there are not cross-connections
        if hasattr(self, "unilateral_system"):
            bilateral_theta = self.get_theta(output="list", order="by_side")
            ipsi_theta = bilateral_theta[:len(self.edges)]
            contra_theta = bilateral_theta[len(self.edges):]
            
            self.unilateral_system.set_theta(ipsi_theta)
            self.A_i = self.unilateral_system.A.copy()
            
            self.unilateral_system.set_theta(contra_theta)
            self.A_c = self.unilateral_system.A.copy()
        else:
            super()._gen_A()
            
            
    def _gen_B(self):
        """Generate the observation matrix for one side of the neck. The other 
        side will use the same (if there are no cross-connections)."""
        
        # if no cross-connections, use the one-sided observation matrix for 
        # both sides...
        if hasattr(self, 'unilateral_system'):
            self.unilateral_system._gen_B()
            self.B = self.unilateral_system.B
            
        # ...otherwise use parent class's method to compute the "full" 
        # observation matrix
        else:
            super()._gen_B()
            
            
    def load_data(self,
                  data: pd.DataFrame,
                  t_stage: List[int] = [1,2,3,4],
                  spsn_dict: Dict[str, List[float]] = {"path": [1., 1.]},
                  mode: str = "HMM"):
        """Convert table of patient diagnoses into two sets of arrays and 
        matrices (ipsi- & contralateral) for fast marginalization.
        
        The likelihood computation produces a square matrix :math:`\\mathbf{M}` 
        of all combinations of possible diagnoses ipsi- & contralaterally. The 
        elements of this matrix are 
        :math:`M_{ij} = P ( \\zeta_i^{\\text{i}}, \\zeta_j^{\\text{c}} )`. This 
        matrix is data independent and we can utilize it by multiplying 
        matrices :math:`\\mathbf{C}^{\\text{i}}` and 
        :math:`\\mathbf{C}^{\\text{c}}` from both sides. This function 
        constructs these matrices that may also allow to marginalize over 
        incomplete diagnoses.
        
        Args:
            data: Table with rows of patients. Must have a two-level 
                :class:`MultiIndex` where the top-level has categories 'Info' 
                and the name of the available diagnostic modalities. Under 
                'Info', the second level is only 'T-stage', while under the 
                modality, the names of the diagnosed lymph node levels are 
                given as the columns.
                
            t_stage: List of T-stages that should be included in the learning 
                process.

            spsn_dict: Dictionary of specificity :math:`s_P` and :math:`s_N` 
                (in that order) for each observational/diagnostic modality.

            mode: ``"HMM"`` for hidden Markov model and ``"BN"`` for Bayesian 
                network.
        """
        if hasattr(self, 'unilateral_system'):
            self.set_modalities(spsn_dict=spsn_dict)
            
            if mode == "HMM":
                C_dict = {}
                
                for stage in t_stage:                
                    # get subset of patients for specific T-stage
                    subset = data.loc[data['Info', 'T-stage']==stage,
                                      self._modality_dict.keys()].values
                    
                    # create a data matrix for each side
                    ipsi_C = np.zeros(shape=(len(subset), len(self.obs_list)), 
                                    dtype=int)
                    contra_C = np.zeros(shape=(len(subset), len(self.obs_list)), 
                                        dtype=int)
                    
                    # find out where to look for ipsi- and where to look for 
                    # contralateral involvement
                    lnlname_list = data.columns.get_loc_level(
                        self._modality_dict, level=1)[1].to_list()
                    ipsi_idx = np.array([], dtype=int)
                    contra_idx = np.array([], dtype=int)
                    for i,lnlname in enumerate(lnlname_list):
                        if "_i" in lnlname:
                            ipsi_idx = np.append(ipsi_idx, i)
                        elif "_c" in lnlname:
                            contra_idx = np.append(contra_idx, i)
                    
                    # loop through the diagnoses in the table
                    for p,patient in enumerate(subset):
                        # split diagnoses into ipsilateral and contralateral
                        ipsi_diag = patient[ipsi_idx]
                        contra_diag = patient[contra_idx]
                        
                        ipsi_tmp = np.zeros_like(self.obs_list[0], dtype=int)
                        contra_tmp = np.zeros_like(self.obs_list[0], dtype=int)
                        for i,obs in enumerate(self.obs_list):
                            # true if non-missing observations match 
                            # (ipsilateral)
                            if np.all(np.equal(obs, ipsi_diag, 
                                            where=~np.isnan(patient), 
                                            out=np.ones_like(patient, 
                                                                dtype=bool))):
                                ipsi_tmp[i] = 1                                                        
                                
                            # true if non-missing observations match 
                            # (contralateral)
                            if np.all(np.equal(obs, contra_diag, 
                                            where=~np.isnan(patient), 
                                            out=np.ones_like(patient, 
                                                                dtype=bool))):
                                contra_tmp[i] = 1
                                
                        # append each patient's marginalization Vector for both 
                        # sides to the matrix. This yields two matrices with 
                        # shapes (number of patients, possible diagnoses)
                        ipsi_C[p] = ipsi_tmp.copy()
                        contra_C[p] = contra_tmp.copy()
                        
                    # collect the two matrices in a dictionary and put that 
                    # under the respective T-stage
                    C_dict[stage] = {"ipsi": ipsi_C.copy(),
                                    "contra": contra_C.copy()}
                    
            elif mode == "BN":
                subset = data[self._modality_dict.keys()].values
                
                # create a data matrix for each side
                ipsi_C = np.zeros(shape=(len(subset), len(self.obs_list)), 
                                    dtype=int)
                contra_C = np.zeros(shape=(len(subset), len(self.obs_list)), 
                                    dtype=int)
                
                # find out where to look for ipsi- and where to look for contra-
                # lateral involvement
                lnlname_list = data.columns.get_loc_level(
                    self._modality_dict, level=1)[1].to_list()
                ipsi_idx = np.array([], dtype=int)
                contra_idx = np.array([], dtype=int)
                for i,lnlname in enumerate(lnlname_list):
                    if "_i" in lnlname:
                        ipsi_idx = np.append(ipsi_idx, i)
                    elif "_c" in lnlname:
                        contra_idx = np.append(contra_idx, i)
                
                # loop through the diagnoses in the table
                for p,patient in enumerate(subset):
                    # split diagnoses into ipsilateral and contralateral
                    ipsi_diag = patient[ipsi_idx]
                    contra_diag = patient[contra_idx]
                    
                    ipsi_tmp = np.zeros_like(self.obs_list[0], dtype=int)
                    contra_tmp = np.zeros_like(self.obs_list[0], dtype=int)
                    for i,obs in enumerate(self.obs_list):
                        # true if non-missing observations match (ipsilateral)
                        if np.all(np.equal(obs, ipsi_diag, 
                                            where=~np.isnan(patient), 
                                            out=np.ones_like(patient, 
                                                            dtype=bool))):
                            ipsi_tmp[i] = 1                                                        
                            
                        # true if non-missing observations match (contralateral)
                        if np.all(np.equal(obs, contra_diag, 
                                            where=~np.isnan(patient), 
                                            out=np.ones_like(patient, 
                                                            dtype=bool))):
                            contra_tmp[i] = 1
                            
                    # append each patient's marginalization Vector for both 
                    # sides to the matrix. This yields two matrices with shapes 
                    # (number of patients, possible diagnoses)
                    ipsi_C[p] = ipsi_tmp.copy()
                    contra_C[p] = contra_tmp.copy()
                    
                self.ipsi_C = ipsi_C.copy()
                self.contra_C = contra_C.copy()
              
        # if the system has cross-connections, revert to the old method, which 
        # is way slower, but it works.  
        else:
            super().load_data(data=data,
                              t_stage=t_stage,
                              spsn_dict=spsn_dict,
                              mode=mode)
            
    # TODO: write new likelihood function