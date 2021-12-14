import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List, Any

import emcee
from emcee
import h5py

from .unilateral import Unilateral
from .bilateral import Bilateral
from .midline import MidlineBilateral


def lyprox_to_lymph(
    data: pd.DataFrame, 
    method: str = "unilateral",
    modalities: List[str] = ["MRI", "PET"],
    convert_t_stage: Optional[Dict[int, Any]] = None
) -> pd.DataFrame:
    """Convert LyProX output into ``DataFrame`` that the lymph package can use 
    for sampling.
    
    Args:
        data: pandas ``DataFrame`` exported from the LyProX interface.
        method: Can be ``"unilateral"``, ``"bilateral"`` or ``"midline"``. It 
            corresponds to the three lymphatic network classes that are 
            implemented in the lymph package.
        modalities: List of diagnostic modalities that should be extracted from 
            the exported data.
        convert_t_stage: For each of the possible T-categories (0, 1, 2, 3, 4) 
            this dictionary holds a key where the corresponding value is the 
            'converted' T-category. For example, if one only wants to 
            differentiate between 'early' and 'late', then that dictionary 
            would look like this:
            
            .. code-block:: python
                
                convert_t_stage = {
                    0: 'early',
                    1: 'early',
                    2: 'early',
                    3: 'late',
                    4: 'late'
                }
    Returns: 
        A converted ``DataFrame`` that can then be used with the lymph package.
    """
    noncentral_data = data.loc[data[("tumor", "1", "side")] != "central"]
    lateralization = noncentral_data[("tumor", "1", "side")]
    t_stage_data = noncentral_data[("tumor", "1", "t_stage")]
    midline_extension_data = noncentral_data[("tumor", "1", "extension")]
    
    diagnostic_data = noncentral_data[modalities].drop(columns=["date"], level=2)
    # copying just to get DataFrae of same structure and with same columns
    sorted_data = diagnostic_data.copy()
    # rename columns for assignment later
    sorted_data = sorted_data.rename(columns={"right": "ipsi", "left": "contra"})
    other_ = {"right": "left", "left": "right"}
    
    for mod in modalities:
        for side in ["left", "right"]:
            ipsi_diagnoses = diagnostic_data.loc[
                lateralization == side, (mod, side)
            ]
            contra_diagnoses = diagnostic_data.loc[
                lateralization == side, (mod, other_[side])
            ]
            sorted_data.loc[lateralization == side, (mod, "ipsi")] = ipsi_diagnoses.values
            sorted_data.loc[lateralization == side, (mod, "contra")] = contra_diagnoses.values
    
    if convert_t_stage is not None:
        sorted_data[("info", "tumor", "t_stage")] = [
            convert_t_stage[t] for t in t_stage_data.values
        ]
    else:
        sorted_data[("info", "tumor", "t_stage")] = t_stage_data
    
    if method == "midline":
        sorted_data[("info", "tumor", "midline_extension")] = midline_extension_data
    elif method == "unilateral":
        sorted_data = sorted_data.drop(columns=["contra"], level=1)
        sorted_data.columns = sorted_data.columns.droplevel(1)
    
    return sorted_data


class EnsembleSampler(emcee.EnsembleSampler):
    """A custom wrapper around emcee's ``EnsembleSampler``.
    """
    
    def __init__(
        self, 
        nwalkers, 
        ndim, 
        log_prob_fn, 
        hdf5_filepath,
        hdf5_groupname='/',
        pool=None, 
        moves=None, 
        args=None, 
        kwargs=None, 
        backend=None, 
        vectorize=False, 
        blobs_dtype=None, 
        parameter_names: Optional[Union[Dict[str, int], List[str]]] = None
    ):
        """Before initializing the sampler, extract the instance of the used 
        lymph system and store the filename of the HDF5 file where the sampler's 
        settings and results are supposed to be stored.
        """
        super().__init__(
            nwalkers, 
            ndim, 
            log_prob_fn, 
            pool=pool, 
            moves=moves, 
            args=args, 
            kwargs=kwargs, 
            backend=backend, 
            vectorize=vectorize, 
            blobs_dtype=blobs_dtype, 
            parameter_names=parameter_names
        )
        
        # extract instance of lymph system used
        self.lymph_system = log_prob_fn.__self__
        
        self.hdf5_filepath = hdf5_filepath
        self.hdf5_groupname = hdf5_groupname
    
    
    @classmethod
    def from_dict(
        cls, 
        kwargs_dict: Dict[str, Any],
        log_prob_fn, 
        pool=None, 
        moves=None, 
        args=None, 
        kwargs=None, 
        backend=None, 
        vectorize=False, 
        blobs_dtype=None, 
        parameter_names: Optional[Union[Dict[str, int], List[str]]] = None
    ):
        """Create an ``EnsembleSampler`` from a dictionary of arguments (e.g. 
        as they might be stored in an HDF5 file).
        """
        nwalkers = kwargs_dict["nwalkers"]
        ndim = kwargs_dict["ndim"]
        nsteps = kwargs_dict["nsteps"]
        burnin = kwargs_dict["burnin"]
        
        # get class that was used
        if (cls_name := kwargs_dict["lymph_sys_cls"]) == "Unilateral":
            lymph_sys_cls = Unilateral
        elif cls_name == "Bilateral":
            lymph_sys_cls = Bilateral
        elif cls_name == "MidlineBilateral":
            lymph_sys_cls = MidlineBilateral
        else:
            raise ValueError(
                "Class name defined in attributes does not match any of the "
                "available classes to model lymphatic spread."
            )
        
        # infer method from extracted class and method name
        log_prob_fn = None
        for attr in dir(lymph_sys_cls):
            is_callable = callable(getattr(lymph_sys_cls, attr))
            is_dunder = attr.startswith("__")
            is_match = attr == kwargs_dict["log_prob_fn"]
            if is_callable and not is_dunder and is_match:
                log_prob_fn = getattr(lymph_sys_cls, attr)
        
        if log_prob_fn == None:
            raise ValueError(
                "The specified name of the log-likelihood function is not a "
                "method of the described class."
            )
        
        # instantiate ensemble sampler from extracted values
        ensemble_sampler = cls.__init__(
            # extracted from the provided dictionary
            nwalkers=nwalkers,
            ndim=ndim,
            log_prob_fn=log_prob_fn,
            # provided via keyword arguments
            pool=pool, 
            moves=moves, 
            args=args, 
            kwargs=kwargs, 
            backend=backend, 
            vectorize=vectorize, 
            blobs_dtype=blobs_dtype, 
            parameter_names=parameter_names
        )
        # add final settings
        ensemble_sampler.nsteps = nsteps
        ensemble_sampler.burnin = burnin
    
    def to_dict(
        self,
        keys: List[str] = ["nwalkers", 
                           "ndim", 
                           "nsteps", 
                           "burnin", 
                           "log_prob_fn"]
    ) -> Dict[str, Any]:
        """Try to return a dictionary that fully specifies the 
        ``EnsembleSampler``. The returned dictionary contains essentially all 
        the arguments to call the ``__init__`` method.
        """
        res = {}
        for key in keys:
            if hasattr(self, key):
                res[key] = getattr(self, key)
                
        return res
    
    
    def run_mcmc(self, initial_state, nsteps=None, burnin=None, **kwargs):
        """Wrap emcee's ``run_mcmc`` method so that it stores the used settings 
        before and the resulting samples before performing the actual sampling.
        """
        if nsteps is not None:
            self.nsteps = nsteps
        elif not hasattr(self, "nsteps"):
            raise ValueError(
                "If `nsteps` hasn't been set yet, it must be provided as an "
                "argument in this run function."
            )
        
        if burnin is not None:
            self.burnin = burnin
        elif not hasattr(self, "burnin"):
            raise ValueError(
                "If `burnin` hasn't been set yet, it must be provided as an "
                "argument in this run function."
            )
            
        # store settings about sampler & lymph system
        with h5py.File(self.hdf5_filepath, 'a') as hdf5_file:
            hdf5_group = hdf5_file.require_group(self.hdf5_groupname)
            attrs_dict = self.to_dict()
            attrs_dict.update(self.lymph_system.to_dict())
            for key, value in attrs_dict.items():
                hdf5_group.attrs[key] = value
            
            hdf5_patient_data = hdf5_group.require_dataset("patient_data")
            hdf5_patient_data[...] = self.lymph_system.data
        
        res = super().run_mcmc(initial_state, self.nsteps, progress=True, **kwargs)
        
        # store samples etc
        with h5py.File(self.hdf5_filepath, 'a') as hdf5_file:
            hdf5_group = hdf5_file.require_group(self.hdf5_groupname)
            hdf5_group.attrs["acceptance_fraction"] = np.mean(
                self.acceptance_fraction, axis=0
            )
            hdf5_samples = hdf5_group.require_dataset("samples")
            hdf5_samples[...] = self.get_chain(flat=True, discard=self.burnin)
            
            hdf5_log_prob = hdf5_group.require_dataset("log_prob")
            hdf5_log_prob[...] = self.get_log_prob(flat=True, discard=self.burnin)
            
        return res