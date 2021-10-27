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
        b_c^{\\in} = \\alpha \cdot b_i + (1 - \\alpha) \cdot b_c^{\\not\\in}
        
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
        return np.concatenate([self.noext.base_probs, self.alpha_mix])
    
    @base_probs.setter
    def base_probs(self, new_params: np.ndarray):
        """"""
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