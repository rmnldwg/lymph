import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any

from .bilateral import BilateralSystem


class MidexSystem(BilateralSystem):
    """Model a bilateral lymphatic system where an additional risk factor can 
    be provided in the data: Whether or not the primary tumor extended over the 
    mid-sagittal line. 
    
    It is reasonable to assume (and supported by data) that such an extension 
    significantly increases the risk for metastatic spread to the contralateral 
    side of the neck. This subclass attempts to capture this using a simple 
    assumption: We assume that the probability of spread to the contralateral 
    side for patients *with* midline extension is larger than for patients 
    *without* it, but smaller than the probability of spread to the ipsilateral 
    side. Formally:
    
    .. math::
        b_c^{\in} = \alpha \cdot b_i + (1 - \alpha) \cdot b_c^{\not\in}
        
    where :math:`b_c^{\in}` is the probability of spread from the primary tumor 
    to the contralateral side for patients with midline extension, and 
    :math:`b_c^{\not\in}` for patients without. :math:`\alpha` is the linear 
    mixing parameter.
    """
    
    # I think, I 'only' need to overwrite the methods `load_data`, 
    # `log_likelihood` (and all the wrappers for it probably as well) and 
    # `risk`... Or maybe I can do something smart with the `_evolve` method?
    
    # But the basic idea would be to compute the contralateral side two times, 
    # once for the patients with and once for those without midline extension.