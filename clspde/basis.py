import numpy as np


class Basis:
    """
    class of basis functions for decomposition of solution
    """

    def __init__(
        self, num_of_elems: int, type: str = "poly", steps=np.array([]), n_dims: int = 1
    ):
        self.type = type
        self.n = num_of_elems
        self.n_dims = n_dims
        if steps.size == 0:
            self.steps = np.ones(self.n)
        else:
            self.steps = steps

        self.precalculated_mults = np.zeros((self.n_dims, self.n, self.n))
        for i in range(self.n_dims):
            for n in range(self.n):
                for der in range(self.n):
                    self.precalculated_mults[i, n, der] = np.prod(
                        list(range(max(n + 1 - der, 0), n + 1))
                    )

    def eval(self, x, derivative=np.array([]), ravel=False):
        """
        evaluation of n-th basis funcion in x
        """
        derivative = np.array(np.abs(derivative), dtype=int)
        if derivative.size == 0:
            derivative = np.zeros(self.n, dtype=int)

        result = np.zeros((self.n_dims, self.n))
        for i in range(self.n_dims):
            for n in range(self.n):
                mult = self.precalculated_mults[i, n, derivative[i]]
                # print(x)
                result[i, n] = (
                    x[i] ** (max(n - derivative[i], 0))
                    * mult
                    / ((self.steps[i] / 2) ** derivative[i])
                )
        if ravel:
            mat_result = result[0]
            for i in range(1, self.n_dims):
                mat_result = np.outer(mat_result, result[i])
            return mat_result.ravel(order="C")
        else:
            return result
