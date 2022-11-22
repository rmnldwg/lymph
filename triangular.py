"""
In this module, I try to make the matrix multiplication of two recursively upper
diagonal matrices more efficient.
"""
from timeit import default_timer
from typing import Tuple

import numpy as np

from lymph.unilateral import change_base


def gen_states(num: int):
    """Generates the list of states."""
    states = np.zeros(
        shape=(2**num, num), dtype=int
    )
    for i in range(2**num):
        states[i] = [int(digit) for digit in change_base(i, 2, length=num)]

    return states


class RUTMatrix(np.ndarray):
    """
    Implemetation of a recursively upper triangular matrix of shape n x n, where
    n = 2 ** k.
    """
    def __matmul__(self, other):
        if not isinstance(other, RUTMatrix):
            return super().__matmul__(other)

        if self.shape != other.shape:
            raise ValueError("Shape mismatch.")

        if self.shape == (1,1):
            return super().__matmul__(other)

        l = self.shape[0] // 2
        lower_left = np.zeros(shape=(l,l))
        upper_left = self.quadrant("upper left") @ other.quadrant("upper left")
        lower_right = self.quadrant("lower right") @ other.quadrant("lower right")
        upper_right = (
            self.quadrant("upper left") @ other.quadrant("upper right")
            + self.quadrant("upper right") @ other.quadrant("lower right")
        )
        return np.concatenate([
            np.concatenate([upper_left, upper_right], axis=1),
            np.concatenate([lower_left, lower_right], axis=1)
        ], axis=0)


    def quadrant(self, corner: str):
        """Get one of the four quadrants of the matrix."""
        l = self.shape[0] // 2
        if corner == "upper left":
            return self[:l,:l]
        if corner == "lower right":
            return self[l:,l:]
        if corner == "upper right":
            return self[:l,l:]
        if corner == "lower left":
            return self[l:,:l]

        raise ValueError("Specify one of the quadrants")

    
def measure_multiplication(
    method: str,
    power: int,
    seed: int = 42
) -> Tuple[np.ndarray,float]:
    """
    Measure the multiplication of two recursively upper triangular matrices of
    shape (2^power x 2^power). Once using the "numpy" `method`, and once using "mine".
    """
    rng = np.random.default_rng(seed)

    states = gen_states(power)
    A = rng.uniform(size=(2**power,2**power))
    B = rng.uniform(size=(2**power,2**power))

    if method == "mine":
        A = A.view(RUTMatrix)
        B = B.view(RUTMatrix)

    for i,i_state in enumerate(states):
        for j,j_state in enumerate(states):
            if np.any(np.less(j_state, i_state)):
                A[i,j] = 0.
                B[i,j] = 0.

    start = default_timer()
    result = A @ B
    end = default_timer()

    return result, end - start


if __name__ == "__main__":
    numpy_result, numpy_time = measure_multiplication(method="numpy", power=10)
    mine_result, mine_time = measure_multiplication(method="mine", power=10)

    assert np.all(np.isclose(numpy_result, mine_result))
    print(f"Numpy took {numpy_time} s")
    print(f"Mine took {mine_time} s")
