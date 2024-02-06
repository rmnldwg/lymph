"""
Module containing supporting classes and functions used accross the project.
"""
import warnings
from collections import UserDict
from functools import cached_property, lru_cache, wraps
from typing import Any, Callable

import numpy as np
from cachetools import LRUCache
from pandas._libs.missing import NAType

PatternType = dict[str, bool | NAType | None]
"""Type alias for an involvement pattern."""

DiagnoseType = dict[str, PatternType]
"""Type alias for a diagnose, which is a involvement pattern per diagnostic modality."""


class DelegatorMixin:
    """Mixin class that allows the delegation of attributes from another object."""
    def __init__(self):
        self._delegated = {}


    def init_delegation(self, **from_to) -> None:
        """Initialize the delegation of attributes.

        For each keyword argument that is an attribute of ``self``, the value is a
        list of attributes to delegate to ``self``.

        Inspiration from this came from the `delegation pattern`_.

        .. _delegation pattern: https://github.com/faif/python-patterns/blob/master/patterns/fundamental/delegation_pattern.py

        Example:

        >>> class Delegate:
        ...     def __init__(self):
        ...         self.fancy_attr = "foo"
        ...     @property
        ...     def property_attr(self):
        ...         return "bar"
        ...     @cached_property
        ...     def cached_attr(self):
        ...         return "baz"
        >>> class A(DelegatorMixin):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.delegated = "hello world"
        ...         self.also_delegated = Delegate()
        ...         self.normal_attr = 42
        ...         self.init_delegation(
        ...             delegated=["count"],
        ...             also_delegated=["fancy_attr", "property_attr", "cached_attr"],
        ...         )
        >>> a = A()
        >>> a.delegated.count("l")
        3
        >>> a.count("l")
        3
        >>> a.also_delegated.fancy_attr
        'foo'
        >>> a.fancy_attr
        'foo'
        >>> a.also_delegated.property_attr
        'bar'
        >>> a.property_attr
        'bar'
        >>> a.also_delegated.cached_attr
        'baz'
        >>> a.cached_attr
        'baz'
        >>> a.normal_attr
        42
        >>> a.non_existent
        Traceback (most recent call last):
        ...
        AttributeError: 'A' object has no attribute 'non_existent'
        """
        for attr, sub_attrs in from_to.items():
            attr_obj = getattr(self, attr)

            for sub_attr in sub_attrs:
                if not hasattr(attr_obj, sub_attr):
                    raise AttributeError(
                        f"Attribute '{sub_attr}' not found in '{attr_obj}'"
                    )

                if sub_attr in self._delegated:
                    warnings.warn(
                        f"Attribute '{sub_attr}' already delegated. Overwriting."
                    )
                self._delegated[sub_attr] = (attr_obj, sub_attr)

    def __getattr__(self, name):
        if name in self._delegated:
            attr = getattr(*self._delegated[name])

            if not callable(attr):
                return attr

            @wraps(attr)
            def wrapper(*args, **kwargs):
                return attr(*args, **kwargs)

            return wrapper

        return super().__getattribute__(name)


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
    """Private method that checks whether specificity and sensitvity
    are valid.

    Args:
        spsn (list): list with specificity and sensiticity

    Raises:
        ValueError: raises a value error if the spec or sens is not a number btw. 0.5 and 1.0
    """
    has_len_2 = len(spsn) == 2
    is_above_lb = np.all(np.greater_equal(spsn, 0.5))
    is_below_ub = np.all(np.less_equal(spsn, 1.))
    if not has_len_2 or not is_above_lb or not is_below_ub:
        msg = ("For each modality provide a list of two decimals "
            "between 0.5 and 1.0 as specificity & sensitivity "
            "respectively.")
        raise ValueError(msg)


def change_base(
    number: int,
    base: int,
    reverse: bool = False,
    length: int | None = None
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

    convert_string = "0123456789ABCDEF"
    result = ''

    if number == 0:
        result += '0'
    else:
        while number >= base:
            result += convert_string[number % base]
            number = number//base
        if number > 0:
            result += convert_string[number]

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


def check_modality(modality: str, spsn: list):
    """Private method that checks whether all inserted values
    are valid for a confusion matrix.

    Args:
        modality (str): name of the modality
        spsn (list): list with specificity and sensiticity

    Raises:
        TypeError: returns a type error if the modality is not a string
        ValueError: raises a value error if the spec or sens is not a number btw. 0.5 and 1.0
    """
    if not isinstance(modality, str):
        raise TypeError("Modality names must be strings.")

    has_len_2 = len(spsn) == 2
    is_above_lb = np.all(np.greater_equal(spsn, 0.5))
    is_below_ub = np.all(np.less_equal(spsn, 1.))

    if not has_len_2 or not is_above_lb or not is_below_ub:
        raise ValueError(
            "For each modality provide a list of two decimals between 0.5 and 1.0 "
            "as specificity & sensitivity respectively."
        )


def clinical(spsn: list) -> np.ndarray:
    """produces the confusion matrix of a clinical modality, i.e. a modality
    that can not detect microscopic metastases

    Args:
        spsn (list): list with specificity and sensitivity of modality

    Returns:
        np.ndarray: confusion matrix of modality
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
    """produces the confusion matrix of a pathological modality, i.e. a modality
    that can detect microscopic metastases

    Args:
        spsn (list): list with specificity and sensitivity of modality

    Returns:
        np.ndarray: confusion matrix of modality
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

    Example:

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

    Example:

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

    Example:

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
    """Method decorator that runs instance's ``trigger()`` when the method is called."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for callback in self.trigger_callbacks:
            callback()
        return result
    return wrapper


if __name__ == "__main__":
    import doctest
    doctest.testmod()


class AbstractLookupDict(UserDict):
    """Abstract ``UserDict`` subclass that can lazily and dynamically return values.

    This class is meant to be subclassed. If one wants to use the functionality
    of lazy and dynamic value retrieval, the subclass must implement a ``__missing__``
    method that returns the value for the given key and raises a ``KeyError`` if
    the value for a key cannot be computed.
    """
    trigger_callbacks: list[Callable]

    def __init__(self, dict=None, /, trigger_callbacks=None, **kwargs):
        """Use keyword arguments to set attributes of the instance.

        In contrast to the default ``UserDict`` constructor, this one instantiates
        any keyword arguments as attributes of the instance and does not put them
        into the dictionary itself.
        """
        super().__init__(dict)

        if trigger_callbacks is None:
            trigger_callbacks = []

        self.trigger_callbacks = trigger_callbacks

        for attr_name, attr_value in kwargs.items():
            if hasattr(self, attr_name):
                raise AttributeError("Cannot set attribute that already exists.")
            setattr(self, attr_name, attr_value)


    def __contains__(self, key: object) -> bool:
        """This exists to trigger ``__missing__`` when checking ``is in``."""
        if super().__contains__(key):
            return True

        if hasattr(self.__class__, "__missing__"):
            try:
                self.__class__.__missing__(self, key)
                return True
            except KeyError:
                return False

        return False


    def clear_without_trigger(self) -> None:
        """Clear the dictionary without triggering the callbacks."""
        self.__dict__["data"].clear()

    def update_without_trigger(self, other=(), /, **kwargs):
        """Update the dictionary without triggering the callbacks."""
        self.__dict__["data"].update(other, **kwargs)


class smart_updating_dict_cached_property(cached_property):
    """Allows setting/deleting dict-like attrs by updating/clearing them."""
    def __set__(self, instance: object, value: Any) -> None:
        dict_like = self.__get__(instance)
        dict_like.clear()
        dict_like.update(value)

    def __delete__(self, instance: object) -> None:
        dict_like = self.__get__(instance)
        dict_like.clear()


def arg0_cache(maxsize: int = 128, cache_class = LRUCache) -> callable:
    """Cache a function only based on its first argument.

    One may choose which ``cache_class`` to use. This will be created with the
    argument ``maxsize``.

    Note:
        The first argument is not passed on to the decorated function. It is basically
        used as a key for the cache and it trusts the user to be sure that this is
        sufficient.
    """
    def decorator(func: callable) -> callable:
        cache = cache_class(maxsize=maxsize)

        @wraps(func)
        def wrapper(arg0, *args, **kwargs):
            if arg0 not in cache:
                cache[arg0] = func(*args, **kwargs)
            return cache[arg0]

        return wrapper

    return decorator
