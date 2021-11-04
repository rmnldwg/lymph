import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any

from .unilateral import Unilateral
from .bilateral import Bilateral


class MidlineBilateral(object):
    """Model a bilateral lymphatic system where an additional risk factor can 
    be provided in the data: Whether or not the primary tumor extended over the 
    mid-sagittal line. 
    
    It is reasonable to assume (and supported by data) that such an extension 
    significantly increases the risk for metastatic spread to the contralateral 
    side of the neck. This class attempts to capture this using a simple 
    assumption: We assume that the probability of spread to the contralateral 
    side for patients *with* midline extension is larger than for patients 
    *without* it, but smaller than the probability of spread to the ipsilateral 
    side. Formally:
    
    .. math::
        b_c^{\\in} = \\alpha \\cdot b_i + (1 - \\alpha) \\cdot b_c^{\\not\\in}
        
    where :math:`b_c^{\\in}` is the probability of spread from the primary tumor 
    to the contralateral side for patients with midline extension, and 
    :math:`b_c^{\\not\\in}` for patients without. :math:`\\alpha` is the linear 
    mixing parameter.
    """
    
    def __init__(
        self, 
        graph: dict = {},
        alpha_mix: float = 0.,
        trans_symmetric: bool = True
    ):
        """The class is constructed in a similar fashion to the 
        :class:`Bilateral`: That class contains one :class:`Unilateral` for 
        each side of the neck, while this class will contain two instances of 
        :class:`Bilateral`, one for the case of a midline extension and one for 
        the case of no midline extension.
        
        Args:
            graph: Dictionary of the same kind as for initialization of 
                :class:`System`. This graph will be passed to the constructors of 
                two :class:`System` attributes of this class.
            alpha_mix: Initial mixing parameter between ipsi- & contralateral 
                base probabilities that determines the contralateral base 
                probabilities for the patients with mid-sagittal extension.
            trans_symmetric: If ``True``, the spread probabilities among the 
                LNLs will be set symmetrically.
        
        See Also:
            :class:`Bilateral`: Two of these are held as attributes by this 
            class. One for the case of a mid-sagittal extension of the primary 
            tumor and one for the case of no such extension.
        """
        self.ext   = Bilateral(
            graph=graph, base_symmetric=False, trans_symmetric=trans_symmetric
        )
        self.noext = Bilateral(
            graph=graph, base_symmetric=False, trans_symmetric=trans_symmetric
        )
        self.alpha_mix = alpha_mix
    
    
    @property
    def base_probs(self) -> np.ndarray:
        """Base probabilities of metastatic lymphatic spread from the tumor(s) 
        to the lymph node levels. This will return a concatenation of the 
        ipsilateral base probabilities and the contralateral ones without the 
        midline extension, as well as - lastly - the mixing parameter alpha.
        
        When setting these, one also needs to provide this mixing parameter as 
        the last entry in the provided array.
        """
        return np.concatenate([self.noext.base_probs, [self.alpha_mix]])
    
    @base_probs.setter
    def base_probs(self, new_params: np.ndarray):
        """Set the base probabilities from the tumor(s) to the LNLs, accounting 
        for the mixing parameter :math:`\\alpha``.
        """
        new_base_probs = new_params[:-1]
        self.alpha_mix = new_params[-1]
        
        # base probabilities for lateralized cases
        self.noext.base_probs = new_base_probs
        
        # base probabilities for cases with tumors extending over the midline
        self.ext.ipsi.base_probs = self.noext.ipsi.base_probs
        self.ext.contra.base_probs = (
            self.alpha_mix * self.noext.ipsi.base_probs 
            + (1 - self.alpha_mix) * self.noext.contra.base_probs
        )
    
    
    @property
    def trans_probs(self) -> np.ndarray:
        """Probabilities of lymphatic spread among the lymph node levels. They 
        are assumed to be symmetric ipsi- & contralaterally by default.
        """
        return self.noext.trans_probs
    
    @trans_probs.setter
    def trans_probs(self, new_params: np.ndarray):
        """Set the new spread probabilities for lymphatic spread from among the 
        LNLs.
        """
        self.noext.trans_probs = new_params
        self.ext.trans_probs = new_params
    
    
    @property
    def spread_probs(self) -> np.ndarray:
        """These are the probabilities representing the spread of cancer along 
        lymphatic drainage pathways per timestep.
        
        The returned array here contains the probabilities of spread from the 
        tumor(s) to the ipsilateral LNLs, then the same values for the spread 
        to the contralateral LNLs, after this the spread probabilities among 
        the LNLs (which is assumed to be symmetric ipsi- & contralaterally) and 
        finally the mixing parameter :math:`\\alpha`.
        """
        spread_probs = self.noext.spread_probs
        return np.concatenate([spread_probs, [self.alpha_mix]])
    
    @spread_probs.setter
    def spread_probs(self, new_params: np.ndarray):
        """Set the new spread probabilities and the mixing parameter 
        :math:`\\alpha`.
        """
        num_base_probs = len(self.noext.ipsi.base_edges)
        
        new_base_probs  = new_params[:2*num_base_probs]
        new_trans_probs = new_params[2*num_base_probs:-1]
        alpha_mix = new_params[-1]
        
        self.base_probs = np.concatenate([new_base_probs, [alpha_mix]])
        self.trans_probs = new_trans_probs
    
    
    @property
    def modalities(self):
        """A dictionary containing the specificity :math:`s_P` and sensitivity 
        :math:`s_N` values for each diagnostic modality.
        
        Such a dictionary can also be provided to set this property and compute 
        the observation matrices of all used systems.
        
        See Also:
            :meth:`Bilateral.modalities`: Getting and setting this property in 
            the normal bilateral model.
            
            :meth:`Unilateral.modalities`: Getting and setting :math:`s_P` and 
            :math:`s_N` for a unilateral model.
        """
        return self.noext.modalities
    
    @modalities.setter
    def modalities(self, modality_spsn: Dict[str, List[float]]):
        """Call the respective getter and setter methods of the bilateral 
        components with and without midline extension.
        """
        self.noext.modalities = modality_spsn
        self.ext.modalities = modality_spsn
    
    
    def load_data(
        self, 
        data: pd.DataFrame,
        t_stages: Optional[List[int]] = None,
        modality_spsn: Optional[Dict[str, List[float]]] = None,
        mode = "HMM"
    ):
        """Load data as table of patients with involvement details and convert 
        it into internal representation of a matrix.
        
        Args:
            data: The table with rows of patients and columns of patient and 
                involvement details. The table's header must have three levels 
                that categorize the individual lymph node level's involvement 
                to the corresponding diagnostic modality (first level), the 
                side of the LNL (second level) and finaly the name of the LNL 
                (third level). Additionally, the patient's T-category must be 
                stored under ('info', 'tumor', 't_stage') and whether the tumor 
                extends over the mid-sagittal line should be noted under 
                ('info', 'tumor', 'midline_extension'). So, part of this table 
                could look like this:
                
                +-------------------------------+---------------------+
                |             info              |       MRI           |
                +-------------------------------+----------+----------+ 
                |             tumor             |   ipsi   |  contra  |
                +---------+---------------------+----------+----------+ 
                | t_stage |  midline_extension  |    II    |    II    |
                +=========+=====================+==========+==========+
                | early   | ``True``            | ``True`` | ``None`` |
                +---------+---------------------+----------+----------+
                | late    | ``True``            | ``None`` | ``None`` |
                +---------+---------------------+----------+----------+
                | early   | ``False``           | ``True`` | ``True`` |
                +---------+---------------------+----------+----------+
            
            t_stages: List of T-stages that should be included in the learning 
                process. If ommitted, the list of T-stages is extracted from 
                the :class:`DataFrame`
            modality_spsn: If no diagnostic modalities have been defined yet, 
                this must be provided to build the observation matrix.
        
        See Also:
            :meth:`Bilateral.load_data`: Loads data into a bilateral network by 
            splitting it into ipsi- & contralateral side and passing each to 
            the respective unilateral method (see below).
            
            :meth:`Unilateral.load_data`: Data loading method of the unilateral 
            network.
            
            :meth:`Unilateral._gen_C`: Generate the data matrix from the tables.
        """
        ext_data = data.loc[data[("info", "tumor", "midline_extension")]]
        noext_data = data.loc[~data[("info", "tumor", "midline_extension")]]
        
        self.ext.load_data(
            ext_data, 
            t_stages=t_stages, 
            modality_spsn=modality_spsn, 
            mode=mode
        )
        self.noext.load_data(
            noext_data, 
            t_stages=t_stages, 
            modality_spsn=modality_spsn, 
            mode=mode
        )
            