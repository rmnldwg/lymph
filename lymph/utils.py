"""
Module containing supporting classes and functions used accross the project.
"""
import logging
from functools import cached_property, lru_cache, wraps
from typing import Any, Sequence

import numpy as np

from lymph import types

logger = logging.getLogger(__name__)



def check_unique_names(graph: dict):
    """Check all nodes in ``graph`` have unique names and no duplicate connections."""
    node_name_set = set()
    for (_, node_name), connections in graph.items():
        if isinstance(connections, set):
            raise TypeError("A node's connection list should not be a set (ordering)")
        if len(connections) != len(set(connections)):
            raise ValueError(f"Duplicate connections for node {node_name} in graph")
        if node_name in connections:
            raise ValueError(f"Node {node_name} is connected to itself")

        node_name_set.add(node_name)

    if len(node_name_set) != len(graph):
        raise ValueError("Node names are not unique")


def check_spsn(spsn: list[float]):
    """Check whether specificity and sensitivity are valid."""
    has_len_2 = len(spsn) == 2
    is_above_lb = np.all(np.greater_equal(spsn, 0.5))
    is_below_ub = np.all(np.less_equal(spsn, 1.))
    if not has_len_2 or not is_above_lb or not is_below_ub:
        raise ValueError(
            "For each modality provide a list of two decimals between 0.5 and 1.0 as "
            "specificity & sensitivity respectively."
        )


@lru_cache
def comp_transition_tensor(
    num_parent: int,
    num_child: int,
    is_tumor_spread: bool,
    is_growth: bool,
    spread_prob: float,
    micro_mod: float,
) -> np.ndarray:
    """Compute the transition factors of the edge.

    The returned array is of shape (p,c,c), where p is the number of states of the
    parent node and c is the number of states of the child node.

    Essentially, the tensors computed here contain most of the parametrization of
    the model. They are used to compute the transition matrix.

    This function globally computes and caches the transition tensors, such that we
    do not need to worry about deleting and recomputing them when the parameters of the
    edge change.
    """
    tensor = np.stack([np.eye(num_child)] * num_parent)

    # this should allow edges from trinary nodes to binary nodes
    pad = [0.] * (num_child - 2)

    if is_tumor_spread:
        # NOTE: Here we define how tumors spread to LNLs
        tensor[0, 0, :] = np.array([1. - spread_prob, spread_prob, *pad])
        return tensor

    if is_growth:
        # In the growth case, we can assume that two things:
        # 1. parent and child state are the same
        # 2. the child node is trinary
        tensor[1, 1, :] = np.array([0., (1 - spread_prob), spread_prob])
        return tensor

    if num_parent == 3:
        # NOTE: here we define how the micro_mod affects the spread probability
        micro_spread = spread_prob * micro_mod
        tensor[1,0,:] = np.array([1. - micro_spread, micro_spread, *pad])

        macro_spread = spread_prob
        tensor[2,0,:] = np.array([1. - macro_spread, macro_spread, *pad])

        return tensor

    tensor[1,0,:] = np.array([1. - spread_prob, spread_prob, *pad])
    return tensor


def clinical(spsn: list) -> np.ndarray:
    """Produce the confusion matrix of a clinical modality.

    A clinical modality can by definition *not* detect microscopic metastases.
    """
    check_spsn(spsn)
    sp, sn = spsn
    confusion_matrix = np.array([
        [sp     , 1. - sp],
        [sp     , 1. - sp],
        [1. - sn, sn     ],
    ])
    return confusion_matrix


def pathological(spsn: list) -> np.ndarray:
    """Produce the confusion matrix of a pathological modality.

    A pathological modality can detect microscopic disease, but is unable to
    differentiante between micro- and macroscopic involvement.
    """
    check_spsn(spsn)
    sp, sn = spsn
    confusion_matrix = np.array([
        [sp     , 1. - sp],
        [1. - sn, sn     ],
        [1. - sn, sn     ],
    ])
    return confusion_matrix


def tile_and_repeat(
    mat: np.ndarray,
    tile: tuple[int, int],
    repeat: tuple[int, int],
) -> np.ndarray:
    """Apply the numpy functions `tile`_ and `repeat`_ successively to ``mat``.

    .. _tile: https://numpy.org/doc/stable/reference/generated/numpy.tile.html
    .. _repeat: https://numpy.org/doc/stable/reference/generated/numpy.repeat.html

    >>> mat = np.array([[1, 2], [3, 4]])
    >>> tile_and_repeat(mat, (2, 2), (2, 2))
    array([[1, 1, 2, 2, 1, 1, 2, 2],
           [1, 1, 2, 2, 1, 1, 2, 2],
           [3, 3, 4, 4, 3, 3, 4, 4],
           [3, 3, 4, 4, 3, 3, 4, 4],
           [1, 1, 2, 2, 1, 1, 2, 2],
           [1, 1, 2, 2, 1, 1, 2, 2],
           [3, 3, 4, 4, 3, 3, 4, 4],
           [3, 3, 4, 4, 3, 3, 4, 4]])
    >>> tile_and_repeat(
    ...     mat=np.array([False, True], dtype=bool),
    ...     tile=(1, 2),
    ...     repeat=(1, 3),
    ... )
    array([[False, False, False,  True,  True,  True, False, False, False,
             True,  True,  True]])
    """
    tiled = np.tile(mat, tile)
    repeat_along_0 = np.repeat(tiled, repeat[0], axis=0)
    return np.repeat(repeat_along_0, repeat[1], axis=1)


@lru_cache
def get_state_idx_matrix(lnl_idx: int, num_lnls: int, num_states: int) -> np.ndarray:
    """Return the indices for the transition tensor correpsonding to ``lnl_idx``.

    >>> get_state_idx_matrix(1, 3, 2)
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]])
    >>> get_state_idx_matrix(1, 2, 3)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 2, 2, 2]])
    """
    indices = np.arange(num_states).reshape(num_states, -1)
    block = np.tile(indices, (num_states ** lnl_idx, num_states ** num_lnls))
    return np.repeat(block, num_states ** (num_lnls - lnl_idx - 1), axis=0)


def row_wise_kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the `kronecker product`_ of two matrices row-wise.

    .. _kronecker product: https://en.wikipedia.org/wiki/Kronecker_product

    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[5, 6], [7, 8]])
    >>> row_wise_kron(a, b)
    array([[ 5.,  6., 10., 12.],
           [21., 24., 28., 32.]])
    """
    result = np.zeros((a.shape[0], a.shape[1] * b.shape[1]))
    for i in range(a.shape[0]):
        result[i] = np.kron(a[i], b[i])

    return result


def early_late_mapping(t_stage: int | str) -> str:
    """Map the reported T-category (i.e., 1, 2, 3, 4) to "early" and "late"."""
    t_stage = int(t_stage)

    if 0 <= t_stage <= 2:
        return "early"

    if 3 <= t_stage <= 4:
        return "late"

    raise ValueError(f"Invalid T-stage: {t_stage}")


def trigger(func: callable) -> callable:
    """Decorator that runs instance's ``trigger_callbacks`` when called."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for callback in self.trigger_callbacks:
            callback()
        return result
    return wrapper


class smart_updating_dict_cached_property(cached_property):
    """Allows setting/deleting dict-like attrs by updating/clearing them."""
    def __set__(self, instance: object, value: Any) -> None:
        dict_like = self.__get__(instance)
        dict_like.clear()
        dict_like.update(value)

    def __delete__(self, instance: object) -> None:
        dict_like = self.__get__(instance)
        dict_like.clear()


def dict_to_func(mapping: dict[Any, Any]) -> callable:
    """Transform a dictionary into a function.

    >>> char_map = {'a': 1, 'b': 2, 'c': 3}
    >>> char_map = dict_to_func(char_map)
    >>> char_map('a')
    1
    """
    def callable_mapping(key):
        return mapping[key]

    return callable_mapping


def popfirst(seq: Sequence[Any]) -> tuple[Any, Sequence[Any]]:
    """Return the first element of a sequence and the sequence without it.

    If the sequence is empty, the first element will be ``None`` and the second just
    the empty sequence. Example:

    >>> popfirst([1, 2, 3])
    (1, [2, 3])
    >>> popfirst([])
    (None, [])
    """
    try:
        return seq[0], seq[1:]
    except IndexError:
        return None, seq


def flatten(mapping, parent_key='', sep='_') -> dict:
    """Flatten a nested dictionary.

    >>> flatten({"a": {"b": 1, "c": 2}, "d": 3})
    {'a_b': 1, 'a_c': 2, 'd': 3}
    """
    items = []
    for k, v in mapping.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_and_split(
    mapping: dict,
    expected_keys: list[str],
    sep: str = "_",
) -> tuple[dict, dict]:
    """Unflatten the part of a dict containing ``expected_keys`` and return the rest.

    >>> unflatten_and_split({'a_b': 1, 'a_c_x': 2, 'd_y': 3}, expected_keys=['a'])
    ({'a': {'b': 1, 'c_x': 2}}, {'d_y': 3})
    """
    split_kwargs, global_kwargs = {}, {}
    for key, value in mapping.items():
        left, _, right = key.partition(sep)
        if left not in expected_keys:
            global_kwargs[key] = value
            continue

        tmp = split_kwargs
        if left not in tmp:
            tmp[left] = {}

        tmp = tmp[left]
        tmp[right] = value

    return split_kwargs, global_kwargs


def get_params_from(
    objects: dict[str, types.HasGetParams],
    as_dict: bool = True,
    as_flat: bool = True,
) -> types.ParamsType:
    """Get the parameters from each ``get_params()`` method of the ``objects``."""
    params = {}
    for key, obj in objects.items():
        params[key] = obj.get_params(as_flat=as_flat)

    if as_flat or not as_dict:
        params = flatten(params)

    return params if as_dict else params.values()


def set_params_for(
    objects: dict[str, types.HasSetParams],
    *args: float,
    **kwargs: float,
) -> tuple[float]:
    """Pass arguments to each ``set_params()`` method of the ``objects``."""
    kwargs, global_kwargs = unflatten_and_split(kwargs, expected_keys=objects.keys())

    for key, obj in objects.items():
        obj_kwargs = global_kwargs.copy()
        obj_kwargs.update(kwargs.get(key, {}))
        args = obj.set_params(*args, **obj_kwargs)

    return args


def safe_set_params(
    model: types.ModelT,
    params: types.ParamsType | None = None,
) -> None:
    """Set the ``params`` of the ``model``.

    This infers whether ``params`` is a dict or a list and calls the ``model``'s method
    ``set_params()`` accordingly.
    """
    if params is None:
        return

    if isinstance(params, dict):
        model.set_params(**params)
    else:
        model.set_params(*params)


def synchronize_params(
    get_from: dict[str, types.HasGetParams],
    set_to: dict[str, types.HasSetParams],
) -> None:
    """Get the parameters from one object and set them to another."""
    for key, obj in set_to.items():
        obj.set_params(**get_from[key].get_params(as_dict=True))


def draw_diagnoses(
    diagnose_times: list[int],
    state_evolution: np.ndarray,
    observation_matrix: np.ndarray,
    possible_diagnoses: np.ndarray,
    rng: np.random.Generator | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Given the ``diagnose_times`` and a hidden ``state_evolution``, draw diagnoses."""
    if rng is None:
        rng = np.random.default_rng(seed)

    state_dists_given_time = state_evolution[diagnose_times]
    observation_dists_given_time = state_dists_given_time @ observation_matrix

    drawn_observation_idxs = [
        rng.choice(a=np.arange(len(possible_diagnoses)), p=dist)
        for dist in observation_dists_given_time
    ]
    return possible_diagnoses[drawn_observation_idxs].astype(bool)


def add_or_mult(llh: float, arr: np.ndarray, log: bool = True) -> float:
    """Add or multiply the log-likelihood with the given array."""
    if log:
        return llh + np.sum(np.log(arr))
    return llh * np.prod(arr)
