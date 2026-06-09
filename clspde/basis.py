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


'''
cheb_mat = ((1), 
            (1), 
            (-1,2), 
            (-3,4), 
            (1,-8,8),
            (5,-20,16),
            (-1,18,-48,32))


cheb_mat_diff = ((0), 
                (1), 
                (-1,2), 
                (-3,4), 
                (1,-8,8),
                (5,-20,16),
                (-1,18,-48,32))
'''

class Cheb(Basis):
    """
    class of basis functions for decomposition of solution
    """

    def __init__(
        self, num_of_elems: int, type: str = "poly", steps=np.array([]), n_dims: int = 1
    ):
        super().__init__(num_of_elems, type, steps, n_dims)
        
        self.coefficients = np.zeros((self.n, self.n))
        self.coefficients[0,0] = 1
        self.coefficients[1,1] = 1
        
        for i in range(2, self.n):
            self.coefficients[i] = -self.coefficients[i-2]
            self.coefficients[i, 1:] += 2*self.coefficients[i-1, :-1]
        
        self.precalculated_mults = np.zeros((self.n_dims, self.n, self.n))
        for i in range(self.n_dims):
            for n in range(self.n):
                for der in range(self.n):
                    self.precalculated_mults[i, n, der] = np.prod(
                        list(range(max(n + 1 - der, 0), n + 1))
                    )
    
    def eval_poly(self, dim, poly_num, x, der):
        result = 0
        for monom in range(self.n):
            mult = self.precalculated_mults[dim, monom, der] / ((self.steps[dim] / 2) ** der)
            result += self.coefficients[poly_num, monom] * x ** (max(monom - der, 0)) * mult
        return result
        
    
    def eval(self, x, derivative=np.array([]), ravel=False):
        """
        evaluation of n-th basis funcion in x
        """
        derivative = np.array(np.abs(derivative), dtype=int)
        if derivative.size == 0:
            derivative = np.zeros(self.n, dtype=int)

        result = np.zeros((self.n_dims, self.n))
        for dim in range(self.n_dims):
            for n in range(self.n): #num of elems
                #mult = self.precalculated_mults[i, n, derivative[i]]
                # print(x)
                result[dim, n] = self.eval_poly(dim, n, x[dim], derivative[dim])
        if ravel:
            mat_result = result[0]
            for i in range(1, self.n_dims):
                mat_result = np.outer(mat_result, result[i])
            return mat_result.ravel(order="C")
        else:
            return result