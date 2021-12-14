import numpy as np
from typing import Optional, Union, Dict, List, Any

import emcee
from emcee import ensemble
import h5py

from .unilateral import Unilateral
from .bilateral import Bilateral
from .midline import MidlineBilateral


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