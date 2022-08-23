from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import emcee
import h5py
import numpy as np
import pandas as pd
from scipy.special import factorial as fact

import lymph


def lyprox_to_lymph(
    data: pd.DataFrame,
    method: str = "unilateral",
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
        convert_t_stage: For each of the possible T-categories (0, 1, 2, 3, 4)
            this dictionary holds a key where the corresponding value is the
            'converted' T-category. For example, if one only wants to
            differentiate between 'early' and 'late', then that dictionary
            would look like this (which is also the default):

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
    t_stage_data = data[("tumor", "1", "t_stage")]
    midline_extension_data = data[("tumor", "1", "extension")]

    # Extract modalities
    top_lvl_headers = set(data.columns.get_level_values(0))
    modalities = [h for h in top_lvl_headers if h not in ["tumor", "patient"]]
    diagnostic_data = data[modalities].drop(columns=["date"], level=2)

    if convert_t_stage is None:
        convert_t_stage = {
            0: "early",
            1: "early",
            2: "early",
            3: "late",
            4: "late"
        }
    diagnostic_data[("info", "tumor", "t_stage")] = [
        convert_t_stage[t] for t in t_stage_data.values
    ]

    if method == "midline":
        diagnostic_data[("info", "tumor", "midline_extension")] = midline_extension_data
    elif method == "unilateral":
        diagnostic_data = diagnostic_data.drop(columns=["contra"], level=1)
        diagnostic_data.columns = diagnostic_data.columns.droplevel(1)

    return diagnostic_data


class EnsembleSampler(emcee.EnsembleSampler):
    """A custom wrapper of emcee's ``EnsembleSampler`` that adds a sampling
    method that automatically tracks convergence.
    """
    def __init__(
        self,
        nwalkers,
        ndim,
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
        """Just define a default mixture of moves.
        """
        if moves is None:
            moves = [
                (emcee.moves.DEMove(),        0.8),
                (emcee.moves.DESnookerMove(), 0.2)
            ]

        super().__init__(
            nwalkers,
            ndim,
            log_prob_fn,
            pool,
            moves,
            args,
            kwargs,
            backend,
            vectorize,
            blobs_dtype,
            parameter_names
        )

    def run_sampling(
        self,
        min_steps: int = 0,
        max_steps: int = 10000,
        check_interval: int = 100,
        trust_threshold: float = 50.,
        rel_acor_threshold: float = 0.05,
        verbose: bool = True,
        random_state: Optional[Tuple[Any]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Extract ``start`` from settings of the sampler and perform sampling
        while monitoring the convergence.

        Args:
            min_steps: Minimum number of sampling steps to perform.
            max_steps: Maximum number of sampling steps to perform.
            check_interval: Number of sampling steps after which to check for
                convergence.
            trust_threshold: The autocorrelation estimate is only trusted when
                it is smaller than the number of samples drawn divided by this
                parameter.
            rel_acor_threshold: The relative change of two consequtive trusted
                autocorrelation estimates must fall below.
            verbose: Show progress during sampling and success at the end.
            random_state: A state of numpy=s random number generator. This is
                passed to emcee to make sampling deterministic. Note that due
                to numerical instabilities (I guess), this will not make any
                sampling round completely deterministic.
            **kwargs: Any other ``kwargs`` are directly passed to the ``sample``
                method.

        Returns:
            A pandas :class:`DataFrame` with the autocorrelation estimates.
        """
        if verbose:
            print("Starting sampling")

        if random_state is not None:
            np.random.set_state(random_state)

        if max_steps < min_steps:
            warnings.warn(
                "Sampling param min_steps is larger than max_steps. Swapping."
            )
            tmp = max_steps
            max_steps = min_steps
            min_steps = tmp

        coords = np.random.uniform(
            low=0., high=1.,
            size=(self.nwalkers, self.ndim)
        )
        start = emcee.State(coords, random_state=np.random.get_state())

        iterations = []
        acor_times = []
        old_acor = np.inf
        idx = 0
        is_converged = False

        for sample in self.sample(
            start, iterations=max_steps, progress=verbose, **kwargs
        ):
            # after `check_interval` number of samples...
            if self.iteration < min_steps or self.iteration % check_interval:
                continue

            # ...compute the autocorrelation time and store it in an array.
            new_acor = self.get_autocorr_time(tol=0)
            iterations.append(self.iteration)
            acor_times.append(np.mean(new_acor))
            idx += 1

            # check convergence based on three criterions:
            # - did it run for at least `min_steps`?
            # - has the acor time crossed the N / `trust_theshold` line?
            # - did the acor time stay stable?
            is_converged = self.iteration >= min_steps
            is_converged &= np.all(new_acor * trust_threshold < self.iteration)
            rel_acor_diff = np.abs(old_acor - new_acor) / new_acor
            is_converged &= np.all(rel_acor_diff < rel_acor_threshold)

            # if it has converged, stop
            if is_converged:
                break

            old_acor = new_acor

        if verbose:
            if is_converged:
                print(f"Sampler converged after {self.iteration} steps")
            else:
                print("Max. number of steps reached")

        return pd.DataFrame(
            np.array([iterations, acor_times]).T,
            columns=["iteration", "acor"]
        )


def tupledict_to_jsondict(dict: Dict[Tuple[str], List[str]]) -> Dict[str, List[str]]:
    """Take a dictionary that has tuples as keys and stringify those keys so
    that it can be serialized to JSON.
    """
    jsondict = {}
    for k, v in dict.items():
        if np.any([',' in s for s in k]):
            raise ValueError("Strings in in key tuple must not contain commas")

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


class HDFMixin(object):
    """Mixin for the :class:`Unilateral`, :class:`Bilateral` and
    :class:`MidlineBilateral` classes to provide the ability to store and load
    settings to and from an HDF5 file.
    """
    graph: Dict[Tuple[str], List[str]]
    patient_data: pd.DataFrame
    modalities: Dict[str, List[float]]

    def to_hdf(
        self,
        filename: str,
        name: str = "",
    ):
        """Store some important settings as well as the loaded data in the
        specified HDF5 file.

        Args:
            filename: Name of or path to HDF5 file.
            name: Name of the group where the info is supposed to be
                stored.
        """
        filename = Path(filename).resolve()

        with h5py.File(filename, 'a') as file:
            group = file.require_group(f"{name}")
            group.attrs["class"] = self.__class__.__name__
            group.attrs["graph"] = json.dumps(tupledict_to_jsondict(self.graph))
            group.attrs["modalities"] = json.dumps(self.modalities)
            group.attrs["base_symmetric"] = getattr(
                self, "base_symmetric", "None"
            )
            group.attrs["trans_symmetric"] = getattr(
                self, "trans_symmetric", "None"
            )

        with pd.HDFStore(filename, 'a') as store:
            store.put(
                key=f"{name}/patient_data",
                value=self.patient_data,
                format="fixed",     # due to MultiIndex this needs to be fixed
                data_columns=None
            )

def system_from_hdf(
    filename: str,
    name: str = "",
    **kwargs
):
    """Create a lymph system instance from the information saved in an HDF5
    file.

    Args:
        filename: Name of the HDF5 file where the info is stored.
        name: Subgroup where to look for the stored settings and data.

    Any other keyword arguments are passed directly to the constructor of the
    respective class.

    Returns:
        An instance of :class:`lymph.Unilateral`, :class:`lymph.Bilateral` or
        :class:`lymph.MidlineBilateral`.
    """
    filename = Path(filename).resolve()
    recover_None = lambda val: val if val != "None" else None

    with h5py.File(filename, 'a') as file:
        group = file.require_group(f"{name}")
        classname = group.attrs["class"]
        graph = jsondict_to_tupledict(json.loads(group.attrs["graph"]))
        modalities = json.loads(group.attrs["modalities"])
        base_symmetric = recover_None(group.attrs["base_symmetric"])
        trans_symmetric = recover_None(group.attrs["trans_symmetric"])

    with pd.HDFStore(filename, 'a') as store:
        patient_data = store.get(f"{name}/patient_data")

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

    new_sys = new_cls(
        graph=graph,
        base_symmetric=base_symmetric,
        trans_symmetric=trans_symmetric,
        **kwargs
    )
    new_sys.modalities = modalities
    new_sys.patient_data = patient_data
    return new_sys


def fast_binomial_pmf(k: int, n: int, p: float):
    """Compute the probability mass function of the binomial distribution.
    """
    q = (1. - p)
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
    if number < 0:
        raise ValueError("Cannot convert negative numbers")
    if base > 16:
        raise ValueError("Base must be 16 or smaller!")
    elif base < 2:
        raise ValueError("There is no unary number system, base must be > 2")

    convertString = "0123456789ABCDEF"
    result = ''

    if number == 0:
        result += '0'
    else:
        while number >= base:
            result += convertString[number % base]
            number = number//base
        if number > 0:
            result += convertString[number]

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
        This function cannot deal with parts of the diagnose being unknown. So
        if, e.g., one level isn't reported for a patient, that row will just be
        ignored.
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


def draw_from_simplex(ndim: int, nsample: int = 1) -> np.ndarray:
    """Draw uniformly from an n-dimensional simplex.

    Args:
        ndim: Dimensionality of simplex to draw from.
        nsample: Number of samples to draw from the simplex.

    Returns:
        A matrix of shape (nsample, ndim) that sums to one along axis 1.
    """
    if ndim < 1:
        raise ValueError("Cannot generate less than 1D samples")
    if nsample < 1:
        raise ValueError("Generating less than one sample doesn't make sense")

    rand = np.random.uniform(size=(nsample, ndim-1))
    unsorted = np.concatenate(
        [np.zeros(shape=(nsample,1)), rand, np.ones(shape=(nsample,1))],
        axis=1
    )
    sorted = np.sort(unsorted, axis=1)

    diff_arr = np.concatenate([[-1., 1.], np.zeros(ndim-1)])
    diff_mat = np.array([np.roll(diff_arr, i) for i in range(ndim)]).T
    res = sorted @ diff_mat

    return res