"""
Module containing supporting classes and functions used accross the project.
"""
import logging
import warnings
from collections import UserDict
from functools import cached_property, lru_cache, wraps
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from cachetools import LRUCache

from lymph.types import HasGetParams, HasSetParams

logger = logging.getLogger(__name__)


class DelegationSyncMixin:
    """Mixin to delegate and synchronize an attribute of multiple instances.

    If a container class holds several (i.e. one ore more) instances of a class, this
    mixin can be used with the container class to delegate and synchronize an attribute
    from the instances.

    See the explanation in the :py:class:`DelegatorMixin.init_delegation_sync` method.

    This also works for attributes that are not hashable, such as lists or dictionaries.
    See more details about that in the :py:class:`AccessPassthrough` class docs.
    """
    def __init__(self) -> None:
        self._attrs_to_objects = {}


    def _init_delegation_sync(self, **attrs_to_objects: list[object]) -> None:
        """Initialize the delegation and synchronization of attributes.

        Each keyword argument is the name of an attribute to synchronize. The value
        should be a list of instances for which that attribute should be synchronized.

        Example:

        >>> class Eye:
        ...     def __init__(self, color="blue"):
        ...         self.eye_color = color
        >>> class Person(DelegationSyncMixin):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.left = Eye("green")
        ...         self.right = Eye("brown")
        ...         self._init_delegation_sync(eye_color=[self.left, self.right])
        >>> person = Person()
        >>> person.eye_color        # pop element of sorted set and warn that not synced
        'green'
        >>> person.eye_color = 'red'
        >>> person.left.eye_color == person.right.eye_color == 'red'
        True
        """
        for name, objects in attrs_to_objects.items():
            types = {type(obj) for obj in objects}
            if len(types) > 1:
                raise ValueError(
                    f"Instances of delegated attribute {name} must be of same type"
                )

        self._attrs_to_objects = attrs_to_objects


    def __getattr__(self, name):
        objects = self._attrs_to_objects[name]
        attr_list = [getattr(obj, name) for obj in objects]

        if len(attr_list) == 1:
            return attr_list[0]

        return fuse(attr_list)


    def __setattr__(self, name, value):
        if name != "_attrs_to_objects" and name in self._attrs_to_objects:
            for inst in self._attrs_to_objects[name]:
                setattr(inst, name, value)
        else:
            super().__setattr__(name, value)


class AccessPassthrough:
    """Allows delegated access to an attribute's methods.

    This class is constructed from a list of objects. It allows access to the
    methods and items of the objects in the list. Setting items is also supported, but
    only one level deep.

    It is used by the :py:class:`DelegationSyncMixin` to handle unhashable attributes.
    For example, a delegated and synched attribute might be a dictionary. In this case,
    a call like ``container.attribute["key"]`` would retrieve the right value, but
    setting it via ``container.attribute["key"] = value`` would at best set the value
    on one of the synched instances, but not on all of them. This class handles passing
    the set value to all instances.

    Note:
        This class is not meant to be used directly, but only by the
        :py:class:`DelegationSyncMixin`.

    Below is an example that demonstrates how calls to ``__setitem__``, ``__setattr__``,
    and ``__call__`` are passed through to both instances for which the delegation and
    synchronization is invoked:

    >>> class Param:
    ...     def __init__(self, value):
    ...         self.value = value
    >>> class Model:
    ...     def __init__(self, **kwargs):
    ...         self.params_dict = kwargs
    ...         self.param = Param(sum(kwargs.values()))
    ...     def set_value(self, key, value):
    ...         self.params_dict[key] = value
    >>> class Mixture(DelegationSyncMixin):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.c1 = Model(a=1, b=2)
    ...         self.c2 = Model(a=3, b=4, c=5)
    ...         self._init_delegation_sync(
    ...             params_dict=[self.c1, self.c2],
    ...             param=[self.c1, self.c2],
    ...             set_value=[self.c1, self.c2],
    ...         )
    >>> mixture = Mixture()
    >>> mixture.params_dict["b"]    # get first element and warn that not synced
    4
    >>> mixture.params_dict["a"] = 99
    >>> mixture.c1.params_dict["a"] == mixture.c2.params_dict["a"] == 99
    True
    >>> mixture.param.value
    12
    >>> mixture.param.value = 42
    >>> mixture.c1.param.value == mixture.c2.param.value == 42
    True
    >>> mixture.set_value("c", 100)
    >>> mixture.c1.params_dict["c"] == mixture.c2.params_dict["c"] == 100
    True
    """
    def __init__(self, attr_objects: list[object]) -> None:
        self._attr_objects = attr_objects


    def __getattr__(self, name):
        if len(self._attr_objects) == 1:
            return getattr(self._attr_objects[0], name)

        return fuse([getattr(obj, name) for obj in self._attr_objects])


    def __getitem__(self, key):
        if len(self._attr_objects) == 1:
            return self._attr_objects[0][key]

        return fuse([obj[key] for obj in self._attr_objects])


    def __setattr__(self, name, value):
        if name != "_attr_objects":
            for obj in self._attr_objects:
                setattr(obj, name, value)
        else:
            super().__setattr__(name, value)


    def __setitem__(self, key, value):
        for obj in self._attr_objects:
            obj[key] = value


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return_values = []
        for obj in self._attr_objects:
            return_values.append(obj(*args, **kwds))

        return fuse(return_values)


    def __len__(self) -> int:
        if len(self._attr_objects) == 1:
            return len(self._attr_objects[0])

        return fuse([len(obj) for obj in self._attr_objects])


def fuse(objects: list[Any]) -> Any:
    """Try to fuse ``objects`` and return one result.

    TODO: This should not immediately return an ``AccessPassthrough`` just because the
    ``objects`` are not all equal. It should do so, when the ``objects`` may be dict-
    like or be callables... I need to think of a proper criterion.

    What about return an ``AccessPassthrough`` when the type is one of those defined
    in this package?
    """
    if all(objects[0] == obj for obj in objects[1:]):
        return objects[0]

    try:
        return sorted(set(objects)).pop()
    except TypeError:
        return AccessPassthrough(objects)


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
    """Decorator that runs instance's ``trigger_callbacks`` when called."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for callback in self.trigger_callbacks:
            callback()
        return result
    return wrapper


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

    Example:

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

    Example:

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


def set_params_for(
    objects: dict[str, HasSetParams],
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


def synchronize_params(
    get_from: dict[str, HasGetParams],
    set_to: dict[str, HasSetParams],
) -> None:
    """Get the parameters from one object and set them to another."""
    for key, obj in set_to.items():
        obj.set_params(**get_from[key].get_params(as_dict=True))


def set_bilateral_params_for(
    *args: float,
    ipsi_objects: dict[str, HasSetParams],
    contra_objects: dict[str, HasSetParams],
    is_symmetric: bool = False,
    **kwargs: float,
) -> tuple[float]:
    """Pass arguments to each ``set_params()`` method of the ``objects``."""
    kwargs, global_kwargs = unflatten_and_split(kwargs, expected_keys=["ipsi", "contra"])

    ipsi_kwargs = global_kwargs.copy()
    ipsi_kwargs.update(kwargs.get("ipsi", {}))
    args = set_params_for(ipsi_objects, *args, **ipsi_kwargs)

    if is_symmetric:
        synchronize_params(ipsi_objects, contra_objects)
    else:
        contra_kwargs = global_kwargs.copy()
        contra_kwargs.update(kwargs.get("contra", {}))
        args = set_params_for(contra_objects, *args, **contra_kwargs)

    return args


def has_any_dunder_method(obj: Any, *methods: str) -> bool:
    """Check whether a class has any of the given dunder methods."""
    return any(hasattr(obj, method) for method in methods)


def check_unique_and_get_first(objects: Iterable, attr: str = "") -> Any:
    """Check if ``objects`` are unique via a set and return of them.

    This function is meant to be used with the ``AccessPassthrough`` class. It is
    used to retrieve the last element of a set of values that are not synchronized.
    """
    object_set = set(objects)
    if len(object_set) > 1:
        warnings.warn(f"{attr} not synced: {object_set}. Setting should sync.")
    return sorted(object_set).pop()


if __name__ == "__main__":
    class Param:
        def __init__(self, value):
            self.value = value
    class Model:
        def __init__(self, **kwargs):
            self.params_dict = kwargs
            self.param = Param(sum(kwargs.values()))
        def set_value(self, key, value):
            self.params_dict[key] = value
    class Mixture(DelegationSyncMixin):
        def __init__(self):
            super().__init__()
            self.c1 = Model(a=1, b=2)
            self.c2 = Model(a=3, b=4, c=5)
            self._init_delegation_sync(
                params_dict=[self.c1, self.c2],
                param=[self.c1, self.c2],
                set_value=[self.c1, self.c2],
            )
    mixture = Mixture()
    mixture.params_dict["b"]
    mixture.params_dict["a"] = 99
    mixture.param.value = 42
    mixture.set_value("c", 100)
