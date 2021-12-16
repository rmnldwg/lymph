from h5py._hl import dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Optional, Union, Dict, List, Any, Tuple
import json

import emcee
import h5py

import lymph


def lyprox_to_lymph(
    data: pd.DataFrame, 
    method: str = "unilateral",
    modalities: List[str] = ["MRI", "PET"],
    convert_t_stage: Optional[Dict[int, Any]] = None
) -> pd.DataFrame:
    """Convert LyProX output into pandas :class:`DataFrame` that the lymph 
    package can use for sampling.
    
    Args:
        data: Patient data exported from the LyProX interface.
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
        A converted pandas :class:`DataFrame` that can then be used with the 
        lymph package.
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
    """A custom version of emcee's ``EnsembleSampler`` that adds convenience 
    methods for storing and loading settings, samples and more to and from an 
    HDF5 file for better reproduceability.
    """
    def __init__(
        self, 
        log_prob_fn, 
        ndim, 
        nwalkers=None, 
        pool=None, 
        moves=None, 
        args=None, 
        kwargs=None, 
        backend=None, 
        vectorize=False, 
        blobs_dtype=None, 
        parameter_names: Optional[Union[Dict[str, int], List[str]]] = None
    ):
        """This class' constructor defines two more default arguments: 
        ``nwalkers`` is now just 10 times ``ndim`` if not otherwise specified 
        and ``moves`` is 80% :class:`DEMove` with 20% :class:`DESnookerMove`.
        
        See Also:
            :class:`emcee.EnsembleSampler`: This utility class inherits all its 
            main functionality from `emcee <https://emcee.readthedocs.io>`_.
        """
        if nwalkers is None:
            nwalkers = 10 * ndim
        
        if moves is None:
            moves = [(emcee.moves.DEMove(),        0.8), 
                     (emcee.moves.DESnookerMove(), 0.2)]
        
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
    
    def to_hdf5(
        self, 
        filename: str, 
        groupname: str = "", 
        attr_list: List[str] = ["nwalkers", "ndim", "acceptance_fraction"]
    ):
        """Export samples and important settings to an HDF5 file. 
        
        Args:
            filename: Name of or path to HDF5 file.
            groupname: Name of the group where the info is supposed to be 
                stored. There, a new subgroup 'sampler' will be created which 
                will then hold the attributes of the ``EnsembleSampler`` and 
                the chain.
            attr_list: List of attributes of the ``EnsembleSampler`` to store 
                in the HDF5 file.
        """
        filename = Path(filename).resolve()
        with h5py.File(filename, mode='a') as file:
            group = file.require_group(f"{groupname}/sampler")
            for attr in attr_list:
                if hasattr(self, attr):
                    group.attrs[attr] = getattr(self, attr)
            chain = group.require_dataset("chain")
            chain[...] = self.get_chain()
            log_prob = group.require_dataset("log_prob")
            log_prob[...] = self.get_log_prob()
    
    @classmethod
    def from_hdf5(
        cls,
        log_prob_fn: Callable,
        filename: str, 
        groupname: str = "",
        **kwargs
    ):
        """Create an ``EnsembleSampler`` from some stored parameters and a 
        log-probability function.
        
        Args:
            filename: Name of or path to HDF5 file.
            groupname: Name of the group where the info is supposed to be 
                stored. There, it searches the subgroup 'sampler' for the 
                attributes to create an ``EnsembleSampler`` from.
            log_prob_fn: The log-probability function to use for sampling.
        
        Returns:
            A new ``EnsembleSampler``.
        """
        filename = Path(filename).resolve()
        with h5py.File(filename, 'r') as file:
            group = file[f"{groupname}/sampler"]
            nwalkers = group.attrs["nwalkers"]
            ndim = group.attrs["ndim"]
        
        return cls(
            nwalkers=nwalkers,
            ndim=ndim,
            log_prob_fn=log_prob_fn,
            **kwargs
        )


def tupledict_to_jsondict(dict: Dict[Tuple[str], List[str]]) -> Dict[str, List[str]]:
    """Take a dictionary that has tuples as keys and stringify those keys so 
    that it can be serialized to JSON.
    """
    jsondict = {}
    for k, v in dict.items():
        jsondict[",".join(k)] = v
    return jsondict

def jsondict_to_tupledict(dict: Dict[str, List[str]]) -> Dict[Tuple[str], List[str]]:
    """Take a serialized JSON dictionary where the keys are strings of 
    comma-separated names and convert them into keys of tuples.
    """
    tupledict = {}
    for k, v in dict.items():
        tupledict[tuple(n for n in k.split(","))] = v
    return tupledict
    


class HDF5Mixin(object):
    """Mixin for the :class:`Unilateral`, :class:`Bilateral` and 
    :class:`MidlineBilateral` classes to provide the ability to store and load 
    settings to and from an HDF5 file.
    """
    graph: Dict[Tuple[str], List[str]]
    
    def to_hdf5(
        self,
        filename: str,
        groupname: str = "",
    ):
        """Store some important settings as well as the loaded data in the 
        specified HDF5 file.
        
        Args:
            filename: Name of or path to HDF5 file.
            groupname: Name of the group where the info is supposed to be 
                stored. There, a new subgroup 'lymph' will be created which 
                will then hold the attributes of the respective class and its 
                data.
        """
        filename = Path(filename).resolve()
        
        with h5py.File(filename, 'a') as file:
            group = file.require_group(f"{groupname}/lymph")
            group.attrs["class"] = self.__class__.__name__
            group.attrs["graph"] = json.dumps(tupledict_to_jsondict(self.graph))
            patient_data = group.require_dataset("patient_data")
            patient_data[...] = self.patient_data
    

def from_hdf5(
    filename: str,
    groupname: str = "",
    **kwargs
):
    """Create a lymph system instance from the information saved in an HDF5 
    file.
    """
    filename = Path(filename).resolve()
    
    with h5py.File(filename, 'a') as file:
        group = file.require_group(f"{groupname}/lymph")
        classname = group.attrs["class"]
        graph = jsondict_to_tupledict(group.attrs["graph"])
        patient_data = group["patient_data"]
    
    if classname == "Unilateral":
        new_cls = lymph.Unilateral
    elif classname == "Bilateral":
        new_cls = lymph.Bilateral
    elif classname == "MidlineBilateral":
        new_cls = lymph.MidlineBilateral
    else:
        raise RuntimeError(
            "The classname loaded from the file does not correspond to an "
            "implemented class in the `lymph` package."
        )
    
    new_sys = new_cls(graph=graph, **kwargs)
    new_sys.patient_data = patient_data
    return new_sys