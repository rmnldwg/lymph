import numpy as np


def are_probabilities(test_array: np.ndarray) -> bool:
    """Check that all entries are within 0 and 1"""
    are_ge_0 = np.all(
        np.greater_equal(test_array, 0.)
    )
    are_le_1 = np.all(
        np.less_equal(test_array, 1.)
    )
    return are_ge_0 and are_le_1