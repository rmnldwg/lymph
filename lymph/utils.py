from __future__ import annotations
import numpy as np
from scipy.special import factorial as fact
import pandas as pd
from pathlib import Path
from typing import Callable, Optional, Union, Dict, List, Any, Tuple
import json
import warnings

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

    `LyProX <https://lyprox.org>`_ is our online interface where we make
    detailed patterns of involvement on a per-patient basis available and
    visualize it in useful ways.

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
    """A custom wrapper of emcee's ``EnsembleSampler`` that adds convenience
    methods for storing and loading settings, samples and more to and from an
    HDF5 file for better reproduceability.
    """
    def __init__(
        self,
        ndim,
        log_prob_fn,
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

    def run_mcmc(self, nsteps, **kwargs):
        """Extract ``initial_state`` from settings of the sampler and call
        parent's ``run_mcmc`` method.
        """
        initial_state = np.random.uniform(
            low=0., high=1.,
            size=(self.nwalkers, self.ndim)
        )
        return super().run_mcmc(initial_state, nsteps, progress=True, **kwargs)

    def to_hdf5(
        self,
        filename: str,
        groupname: str = "",
        overwrite: bool = True,
        attr_list: List[str] = ["nwalkers", "ndim", "acceptance_fraction"]
    ):
        """Export samples and important settings to an HDF5 file.

        Args:
            filename: Name of or path to HDF5 file.
            groupname: Name of the group where the info is supposed to be
                stored. There, a new subgroup 'sampler' will be created which
                will then hold the attributes of the ``EnsembleSampler`` and
                the chain.
            overwrite: If ``True``, any data that might already be stored at
                the location will be deleted and overwritten.
            attr_list: List of attributes of the ``EnsembleSampler`` to store
                in the HDF5 file.
        """
        filename = Path(filename).resolve()
        with h5py.File(filename, mode='a') as file:
            group = file.require_group(f"{groupname}/sampler")
            for attr in attr_list:
                if hasattr(self, attr):
                    group.attrs[attr] = getattr(self, attr)

            chain = self.get_chain()
            log_prob = self.get_log_prob()

            if overwrite:
                try:
                    del group["chain"]
                except KeyError:
                    pass
                try:
                    del group["log_prob"]
                except KeyError:
                    pass
            group.create_dataset("chain", data=chain)
            group.create_dataset("log_prob", data=log_prob)

    @classmethod
    def from_hdf5(
        cls,
        log_prob_fn: Callable,
        filename: str,
        groupname: str = "",
        **kwargs
    ) -> EnsembleSampler:
        """Create an ``EnsembleSampler`` from some stored parameters and a
        log-probability function.

        Args:
            filename: Name of or path to HDF5 file.
            groupname: Name of the group where the info is supposed to be
                stored. There, it searches the subgroup 'sampler' for the
                attributes to create an ``EnsembleSampler`` from.
            log_prob_fn: The log-probability function to use for sampling.

        Returns:
            A new instance with some important settings already loaded from the
            HDF5 file.
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

    @staticmethod
    def get_chain_from_hdf5(filename: str, groupname: str = "") -> np.ndarray:
        """Get the chain that was stored in an HDF5 file previously.
        """
        filename = Path(filename).resolve()

        with h5py.File(filename, 'r') as file:
            group = file[f"{groupname}/sampler"]
            chain = np.array(group["chain"])

        return chain

    @staticmethod
    def get_log_prob_from_hdf5(filename: str, groupname: str = "") -> np.ndarray:
        """Get the log_prob that was stored in an HDF5 file previously.
        """
        filename = Path(filename).resolve()

        with h5py.File(filename, 'r') as file:
            group = file[f"{groupname}/sampler"]
            log_prob = np.array(group["log_prob"])

        return log_prob


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
    patient_data: pd.DataFrame
    modalities: Dict[str, List[float]]

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
            group.attrs["modalities"] = json.dumps(self.modalities)

        with pd.HDFStore(filename, 'a') as store:
            store.put(
                key=f"{groupname}/lymph/patient_data",
                value=self.patient_data,
                format="fixed",     # due to MultiIndex this needs to be fixed
                data_columns=None
            )


def system_from_hdf5(
    filename: str,
    groupname: str = "",
    **kwargs
):
    """Create a lymph system instance from the information saved in an HDF5
    file.

    Args:
        filename: Name of the HDF5 file where the info is stored.
        groupname: Subgroup where to look for the stored settings.

    Any other keyword arguments are passed directly to the constructor of the
    respective class.

    Returns:
        An instance of :class:`lymph.Unilateral`, :class:`lymph.Bilateral` or
        :class:`lymph.MidlineBilateral`.
    """
    filename = Path(filename).resolve()

    with h5py.File(filename, 'a') as file:
        group = file.require_group(f"{groupname}/lymph")
        classname = group.attrs["class"]
        graph = jsondict_to_tupledict(json.loads(group.attrs["graph"]))
        modalities = json.loads(group.attrs["modalities"])

    with pd.HDFStore(filename, 'a') as store:
        patient_data = store.get(f"{groupname}/lymph/patient_data")

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
    new_sys.modalities = modalities
    new_sys.patient_data = patient_data
    return new_sys


def fast_binomial_pmf(k, n, p):
    """
    Compute the probability mass function of the binomial distribution.
    """
    q = (1 - p)
    binom_coeff = fact(n) / (fact(k) * fact(n - k))
    return binom_coeff * p**k * q**(n - k)


def change_base(
    number: int,
    base: int,
    reverse: bool = False,
    length: Optional[int] = None
) -> str:
    """Convert an integer into another base.

    Args:
        number: Number to convert
        base: Base of the resulting converted number
        reverse: If true, the converted number will be printed in reverse order.
        length: Length of the returned string. If longer than would be
            necessary, the output will be padded.

    Returns:
        The (padded) string of the converted number.
    """

    if base > 16:
        raise ValueError("Base must be 16 or smaller!")

    convertString = "0123456789ABCDEF"
    result = ''
    while number >= base:
        result = result + convertString[number % base]
        number = number//base
    if number > 0:
        result = result + convertString[number]

    if length is None:
        length = len(result)
    elif length < len(result):
        length = len(result)
        warnings.warn("Length cannot be shorter than converted number.")

    pad = '0' * (length - len(result))

    if reverse:
        return result + pad
    else:
        return pad + result[::-1]


def comp_state_dist(table: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute the distribution of distinct states/diagnoses from a table of
    individual diagnoses detailing the patterns of lymphatic progression per
    patient.

    Args:
        table: Rows of patients and columns of LNLs, reporting which LNL was
            involved for which patient.

    Returns:
        A histogram of unique states and a list of the corresponding state
        labels.

    Note:
        This, in contrast to :meth:`Unilateral._gen_C`, cannot deal with parts
        of the diagnose being unknown. So if, e.g., one level isn't reported
        for a patient, that row will just be ignored.
    """
    _, num_cols = table.shape
    table = table.astype(float)
    state_dist = np.zeros(shape=2**num_cols, dtype=int)
    for row in table:
        if not np.any(np.isnan(row)):
            idx = int(np.sum([n * 2**i for i,n in enumerate(row[::-1])]))
            state_dist[idx] += 1

    state_labels = []
    for i in range(2**num_cols):
        state_labels.append(change_base(i, 2, length=num_cols))

    return state_dist, state_labels


def draw_diagnose_times(
    num_patients: int,
    stage_dist: Dict[Any, float],
    diag_times: Optional[Dict[Any, int]] = None,
    time_dists: Optional[Dict[Any, List[float]]] = None,
) -> Tuple[List[int], List[Any]]:
    """Draw T-stages from a distribution over them and determine the
    corresponding diagnose time or draw a one from a distribution over diagnose
    times defined for the respective T-stage.

    Args:
        num_patients: Number of patients to draw diagnose times for.
        stage_dist: Distribution over T-stages.
        diag_times: Fixed diagnose time for a given T-stage.
        time_dists: Holds a distribution over diagnose times for each T-stage
            from which the diagnose times will be drawn if it is given. If this
            is ``None``, ``diag_times`` must be provided.

    Returns:
        The drawn T-stages as well as the drawn diagnose times.
    """
    if not np.isclose(np.sum(stage_dist), 1):
        raise ValueError("Distribution over T-stages must sum to 1.")

    # draw the diagnose times for each patient
    if diag_times is not None:
        t_stages = list(diag_times.keys())
        drawn_t_stages = np.random.choice(
            t_stages,
            p=stage_dist,
            size=num_patients
        )
        drawn_diag_times = [diag_times[t] for t in drawn_t_stages]

    elif time_dists is not None:
        t_stages = list(time_dists.keys())
        max_t = len(time_dists[t_stages[0]]) - 1
        time_steps = np.arange(max_t + 1)

        drawn_t_stages = np.random.choice(
            t_stages,
            p=stage_dist,
            size=num_patients
        )
        drawn_diag_times = [
            np.random.choice(time_steps, p=time_dists[t])
            for t in drawn_t_stages
        ]

    else:
        raise ValueError(
            "Either `diag_times`or `time_dists` must be provided"
        )

    return drawn_t_stages, drawn_diag_times