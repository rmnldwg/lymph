"""
In this module, I try to make the matrix multiplication of two recursively upper
diagonal matrices more efficient.
"""
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


if __name__ == "__main__":
    num = 2
    states = gen_states(num)
    A = np.random.uniform(size=(2**num,2**num)).view(RUTMatrix)
    B = np.random.uniform(size=(2**num,2**num)).view(RUTMatrix)

    for i,i_state in enumerate(states):
        for j,j_state in enumerate(states):
            if np.any(np.less(j_state, i_state)):
                A[i,j] = 0.
                B[i,j] = 0.

    print(A @ B - A.view(np.ndarray) @ B.view(np.ndarray))
