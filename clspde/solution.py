import numpy as np
import copy
import itertools

import utils
from utils import dir

from qr_solver import QR_solve, SVD_solve
from basis import Basis

class Solution:
    def __init__(
        self,
        #basis,
        n_dims: int,
        dim_sizes: np.array,
        area_lims: np.array,
        power: int,
        n_funcs: int = 1,
        periodic = None
    ) -> None:
        """
        initiation of solution
        init of grid of cells, initial coefs, basis
        area_lims - np.array of shape (n_dims, 2) - pairs of upper and lower limits of corresponding dim
        dim_sizes - np. array of shape (n_dims)
        """

        self.area_lims = np.array(area_lims)
        self.n_dims = n_dims  # = len(dim_sizes)
        self.dim_sizes = np.array(dim_sizes)  # n of steps for all directions
        self.power = power
        self.n_funcs = n_funcs
        self.init_grid()
        self.steps = (self.area_lims[:, 1] - self.area_lims[:, 0]) / self.dim_sizes
        self.basis = Basis(power, steps=self.steps, n_dims=n_dims)
        self.split_mats_inited = False
        if periodic is None:
            periodic = [False]* self.n_dims
        self.periodic = periodic
        
    def init_grid(self) -> None:
        self.cells_shape = tuple(
            [self.n_funcs] + list(self.dim_sizes) + [self.power] * self.n_dims
        )
        self.cells_coefs = np.ones(self.cells_shape) * 0.4
        self.cell_size = self.n_funcs * (self.power**self.n_dims)

    def cell_index(self, cell_num):
        if self.n_dims == 2:
            cell_ind = (
                cell_num[1] + cell_num[0] * self.dim_sizes[1]
            )  # + cell_num[2] * prod(self.dim_sizes[:2]...
        elif self.n_dims == 1:
            cell_ind = cell_num[0]
        else:
            raise LookupError
        return cell_ind

    def localize(
        self, global_point: np.array, cells_closed_right: bool = False
    ) -> np.array:
        offset = global_point - self.area_lims[:, 0]
        if cells_closed_right:
            shift = np.array(offset % self.steps < 1e-12, dtype=int)
            cell_num = np.array(
                np.floor(offset / self.steps) - shift,
                dtype=int,
            )
        else:
            cell_num = np.array(np.floor(offset / self.steps), dtype=int)
        local_point = 2 * (offset / np.array(self.steps) - np.array(cell_num) - 0.5)

        return cell_num, local_point

    def globalize(self, cell_num: np.array, local_point: np.array) -> np.array:
        global_point = (
            self.area_lims[:, 0]
            + (np.array(local_point) + 2 * np.array(cell_num) + 1) * self.steps / 2
        )
        return global_point

    def init_split_mats(self):
        self.minus_shift = np.zeros([self.power, self.power])
        self.plus_shift = np.zeros([self.power, self.power])
        for i in range(self.power):
            for j in range(i):
                self.minus_shift[i, j] = comb(i, j) * (-2) ** i
        self.plus_shift = np.abs(self.minus_shift)

        self.split_mats_inited = True

    # only for 2d use
    def split_cells(self):
        if not (self.split_mats_inited):
            self.init_split_mats()
        old_coefs = copy.deepcopy(self.cells_coefs)
        inds = [list(range(size)) for size in self.dim_sizes]
        all_old_cells = list(itertools.product(*inds))

        self.dim_sizes = self.dim_sizes * 2

        self.init_grid()
        self.steps = (self.area_lims[:, 1] - self.area_lims[:, 0]) / self.dim_sizes

        for cell in all_old_cells:
            sol.cells_coefs[tuple(cell + np.array([0, 0]))] = (
                np.transpose(self.minus_shift) @ old_coefs[cell] @ self.minus_shift
            )
            sol.cells_coefs[tuple(cell + np.array([1, 0]))] = (
                np.transpose(self.plus_shift) @ old_coefs[cell] @ self.minus_shift
            )
            sol.cells_coefs[tuple(cell + np.array([0, 1]))] = (
                np.transpose(self.minus_shift) @ old_coefs[cell] @ self.plus_shift
            )
            sol.cells_coefs[tuple(cell + np.array([1, 1]))] = (
                np.transpose(self.plus_shift) @ old_coefs[cell] @ self.plus_shift
            )

    def eval(
        self,
        point: np.array,
        der: np.array,
        func: int = 0,
        cell_num=None,
        local=False,
        cells_closed_right: bool = False,
    ) -> float:
        """
        x - np.array(n_dim, float)
        derivatives - np.array(n_dim, int)
        evaluation of solution function with argument x and list of partial derivatives
        """
        if local:
            local_point = point
        else:
            cell_num, local_point = self.localize(point, cells_closed_right)
        coefs = self.cells_coefs[
            tuple(np.insert(np.array(cell_num, dtype=int), 0, func))
        ]
        result = copy.deepcopy(coefs)
        # applying coefs tensor to evaled basis in point
        basis_evaled = self.basis.eval(local_point, np.abs(der), ravel=False)
        
        #axes = list(range(len(basis_evaled.shape)))[::-1]
        #print(basis_evaled.shape, result.shape)
        #print((list(zip(axes,axes))))
        #result2 = np.tensordot(result, basis_evaled, axes=(list(zip(axes,axes))))
        for b_e in basis_evaled[::-1]:
            result = result.dot(b_e)
        #print(result - result2)
        return result

    def generate_system(
        self,
        cell_num: np.array,
        points: np.array,
        colloc_ops,
        border_ops,
        connect_ops=[],
    ) -> tuple:
        colloc_points, connect_points, border_points = points

        # default connection
        if len(connect_ops) == 0:
            connect_ops = self.default_connect_ops([1, 1])

        # default colloc points
        if len(colloc_points) == 0:
            colloc_points = utils.f_collocation_points(self.power)

        colloc_mat, colloc_r = self.generate_subsystem(
            colloc_ops, cell_num, colloc_points
        )

        left_borders = cell_num == np.zeros(self.n_dims)
        right_borders = cell_num == (self.dim_sizes - 1)

        left_border_for_use = np.array([
            np.logical_and(point == -1, left_borders).any() for point in border_points
        ])
        right_border_for_use = np.array([
            np.logical_and(point == 1, right_borders).any() for point in border_points
        ])
        border_points_for_use = border_points[
            np.logical_or(left_border_for_use, right_border_for_use)
        ]

        border_mat, border_r = self.generate_subsystem(
            border_ops,
            cell_num,
            border_points_for_use,
        )

        left_connect_for_use = np.array([
            np.logical_and(point == -1, ~left_borders).any() for point in connect_points
        ])
        right_connect_for_use = np.array([
            np.logical_and(point == 1, ~right_borders).any() for point in connect_points
        ])
        connect_points_for_use = connect_points[
            np.logical_or(left_connect_for_use, right_connect_for_use)
        ]

        connect_mat, connect_r = self.generate_subsystem(
            connect_ops,
            cell_num,
            connect_points_for_use,
        )
        connect_weight = 1
        res_mat = utils.concat(utils.concat(colloc_mat, border_mat), connect_mat * connect_weight)
        res_right = utils.concat(utils.concat(colloc_r, border_r), connect_r * connect_weight)

        return res_mat, res_right

    def iterate_cells(self, **kwargs) -> None:
        inds = [list(range(size)) for size in self.dim_sizes]
        all_cells = list(itertools.product(*inds))
        cell_shape = tuple([self.power] * self.n_dims)
        for cell in all_cells:
            mat, right = self.generate_system(cell, **kwargs)
            cell_size = np.prod(cell_shape)
            solution = self._solver(A=mat, b=right, solver="np")
            for i in range(self.n_funcs):
                self.cells_coefs[(i, *cell)] = solution[
                    i * cell_size : (i + 1) * cell_size
                ].reshape(cell_shape)

    def _solver(self, A, b, solver="np", alpha=0, **kwargs):
        if solver == "QR":
            res = QR_solve(A, b)
        elif solver == "np":
            b = np.transpose(A) @ b
            A = np.transpose(A) @ A + np.eye(A.shape[1]) * alpha
            res = np.linalg.solve(A, b)
        elif solver == "SVD":
            res = SVD_solve(A, b, **kwargs)
        else:
            raise np.ERR_DEFAULT
        return res

    def solve(self, threshold=1e-5, max_iter=10000, verbose=False, **kwargs) -> None:
        prev_coefs = copy.deepcopy(self.cells_coefs)
        i = 0
        for i in range(max_iter):
            self.iterate_cells(**kwargs)
            residual = np.max(np.abs((prev_coefs - self.cells_coefs)))
            if verbose:
                print(residual)
            if residual < threshold:
                break
            prev_coefs = copy.deepcopy(self.cells_coefs)
        if verbose:
            print("Iterations to converge: ", i)

    def generate_integral(self, time):
        # generate common line
        n = self.power
        integral_cell = (
            1 / np.array(range(1, n + 1)) * ([2, 0] * int(np.ceil(n / 2)))[:n]
        )
        full_line = np.zeros(np.prod(self.cells_shape))
        a = np.zeros((self.power, self.power))
        a[0] = integral_cell
        cell_line = utils.concat(np.ravel(a), np.ravel(a) * 0)

        inds = [list(range(size)) for size in self.dim_sizes]
        all_cells = list(itertools.product(*inds))
        num_of_vars = self.cell_size  # will work only for 2d
        for cell_num in all_cells:
            if cell_num[0] == time:
                cell_ind = cell_num[1] + cell_num[0] * self.dim_sizes[1]
                full_line[cell_ind * num_of_vars : (cell_ind + 1) * num_of_vars] = (
                    cell_line
                )
        return full_line
        # set time moment
        # iterate over space
        # set common line into cells

    def generate_eq(self, cell_num, left_side_operator, right_side_operator, points):
        """
        basic func for generating equation
        """

        def left_side(operator, cell_num, point: np.array) -> np.array:
            """must return row of coeficient for LSE"""
            loc_point = copy.deepcopy(point)
            global_point = self.globalize(cell_num, loc_point)
            x = copy.deepcopy(global_point)

            def u_bas(der, func=0):
                bas_size = int(self.cell_size / self.n_funcs)
                result = np.zeros(self.n_funcs * bas_size)
                result[func * bas_size : (func + 1) * bas_size] = self.basis.eval(
                    loc_point, der, ravel=True
                )
                return result

            def u_loc(der, func=0):
                eval_kwargs = {
                    "point": loc_point,
                    "der": der,
                    "func": func,
                    "local": True,
                    "cell_num": cell_num,
                }
                try:
                    result = self.eval(**eval_kwargs)
                except IndexError:
                    result = self.eval(cells_closed_right=True, **eval_kwargs)
                return result
            return operator(self, u_loc, u_bas, x, loc_point)

        def right_side(operator, cell_num, point: np.array) -> float:
            global_point = self.globalize(cell_num, point)
            # x = global_point
            loc_point = copy.deepcopy(point)

            def u_loc(der, func_num=0):
                return self.eval(
                    loc_point, der, local=True, cell_num=cell_num, func=func_num
                )  # for linearization purpses

            _dir = dir(loc_point)  # neigh_point = loc_point - 2 * dir(loc_point)

            def u_nei(der, func_num=0):
                return self.eval(
                    loc_point - 2 * _dir,
                    der,
                    local=True,
                    cell_num=cell_num + _dir,
                    func=func_num,
                )

            return operator(self, u_loc, u_nei, global_point, loc_point)  # x

        mat = np.zeros((len(points), self.cell_size))
        r_side = np.zeros((len(points)))
        for i in range(len(points)):
            mat[i] = left_side(left_side_operator, cell_num, points[i])
            r_side[i] = right_side(right_side_operator, cell_num, points[i])
        return mat, r_side

    def generate_subsystem(self, ops, cell_num, points: np.array) -> tuple:
        left_ops, right_ops = ops
        n_points = len(points)
        mat_len = len(left_ops) * n_points
        mat = np.zeros((mat_len, self.cell_size))
        r = np.zeros(mat_len)
        for i in range(len(left_ops)):
            mat[n_points*i:n_points*(i+1)], r[n_points*i:n_points*(i+1)] = self.generate_eq(
                cell_num, left_ops[i], right_ops[i], points
            )
        return mat, r

    def generate_connection_couple(self, left_ops, cell_num, points: np.array) -> tuple:
        # left ops must be a pair of functions
        right_ops = [lambda *_: 0] * len(left_ops[0])

        connect_mat = np.zeros((len(left_ops[0]) * len(points), np.prod(self.cells_coefs.shape)))
        for i, point in enumerate(points):
            first_line, _ = self.generate_subsystem(
                [left_ops[0], right_ops], cell_num, np.array([point])
            )
            neigh = tuple(np.array(cell_num) + dir(point))
            neigh_point = point - 2 * dir(point)
            second_line, _ = self.generate_subsystem(
                [left_ops[1], right_ops], neigh, np.array([neigh_point])
            )

            connect_line = np.zeros((len(left_ops[0]), np.prod(self.cells_coefs.shape)))
            index = self.cell_index(cell_num)
            neigh_index = self.cell_index(neigh)
            connect_line[:, index * self.cell_size : (index + 1) * self.cell_size] = (
                first_line
            )
            connect_line[
                :, neigh_index * self.cell_size : (neigh_index + 1) * self.cell_size
            ] = -second_line

            connect_mat[len(left_ops[0]) * i : len(left_ops[0]) * (i+1)] = connect_line
        return np.array(connect_mat)

    def default_connect_ops(self, weights=[100, 10]):
        k1, k2 = weights
        connect_left_operators = []
        connect_right_operators = []

        def dir(point: np.array) -> np.array:
            direction = (np.abs(point) == 1) * (np.sign(point))
            return np.array(direction, dtype=int)

        for func_num in range(self.n_funcs):
            connect_left_operators += [
                lambda __, _, u_bas, x, x_loc, func_num=func_num: k1
                * u_bas(0 * dir(x_loc), func_num),
                lambda __, _, u_bas, x, x_loc, func_num=func_num: k2
                * u_bas(dir(x_loc), func_num),
                ]
            connect_right_operators += [
                lambda __, _, u_bas, x, x_loc, func_num=func_num: k1
                * u_bas(0 * dir(x_loc), func_num),
                lambda __, _, u_bas, x, x_loc, func_num=func_num: k2
                * u_bas(dir(x_loc), func_num),
                ]
        connect_ops = [connect_left_operators, connect_right_operators]
        return connect_ops

    def generate_global_system(
        self, points: np.array, colloc_ops, border_ops, connect_ops=[], connect_weight=1
    ) -> tuple:
        colloc_points, connect_points, border_points = points

        # default connection
        if len(connect_ops) == 0:
            connect_ops = self.default_connect_ops()

        # default colloc points
        if len(colloc_points) == 0:
            colloc_points = utils.f_collocation_points(self.power)

        inds = [list(range(size)) for size in self.dim_sizes]
        all_cells = list(itertools.product(*inds))
        num_of_cells = len(all_cells)
        num_of_vars = self.cell_size  
        
        def generate_global_condition_mat(points, ops, points_filter, _type=None):
            num_of_lines = len(points) * len(ops[0])
            num_of_eqs = len(all_cells) * num_of_lines
            
            global_mat = np.zeros((num_of_eqs, num_of_vars * num_of_cells))
            global_right = np.zeros(num_of_eqs)
            
            for cell_num in all_cells:
                points_for_use = points_filter(points, cell_num)
                cell_ind = self.cell_index(cell_num)
                slice0 = lambda x: slice(cell_ind * num_of_lines, cell_ind * num_of_lines + x, None) 
                slice1 = slice(cell_ind * num_of_vars, (cell_ind + 1) * num_of_vars, None)
                if _type is None or _type == 'default':
                    _mat, _r = self.generate_subsystem(ops, cell_num, points_for_use)
                    global_mat[slice0(_mat.shape[0]), slice1] = _mat
                    global_right[slice0(_mat.shape[0])] = _r
                elif _type=='connect':
                    connect_left_operators, connect_right_operators = ops
                    _mat = self.generate_connection_couple(ops,cell_num,points_for_use)
                    global_mat[slice0(_mat.shape[0])] = _mat
            return global_mat, global_right

        global_colloc_mat, global_colloc_r = generate_global_condition_mat(colloc_points, colloc_ops, self.colloc_points_filter)
        global_border_mat, global_border_r = generate_global_condition_mat(border_points, border_ops, self.border_points_filter)
        global_connect_mat, global_connect_r = generate_global_condition_mat(connect_points, connect_ops, self.connect_points_filter, _type='connect')
        
        res_mat = utils.concat(
            utils.concat(global_colloc_mat, global_border_mat),
            global_connect_mat * connect_weight,
        )
        res_right = utils.concat(
            utils.concat(global_colloc_r, global_border_r),
            global_connect_r * connect_weight,
        )

        notnull_ind = np.sum(res_mat != 0, axis=1) != 0
        res_mat = res_mat[notnull_ind]
        res_right = res_right[notnull_ind]
        return res_mat, res_right

    def colloc_points_filter(self, points, cell_num):
            return points

    def border_points_filter(self, points, cell_num):
        left_borders = cell_num == np.zeros(self.n_dims)
        right_borders = cell_num == (self.dim_sizes - 1)

        left_border_for_use = np.array([
            np.logical_and(np.abs(point+1) < 1e-5, left_borders).any()
            for point in points
        ])
        right_border_for_use = np.array([
            np.logical_and(np.abs(point-1) < 1e-5, right_borders).any()
            for point in points
        ])
        if np.logical_or(left_border_for_use, right_border_for_use).any():
            border_points_for_use = np.array(points)
        else:
            border_points_for_use = np.array([])
        return border_points_for_use
        
    def connect_points_filter(self, points, cell_num):
        left_borders = cell_num == np.zeros(self.n_dims)
        right_borders = cell_num == (self.dim_sizes - 1)

        left_connect_for_use = np.array([
            np.logical_and(np.abs(point+1) < 1e-5, ~left_borders).any()
            for point in points
        ])
        right_connect_for_use = np.array([
            np.logical_and(np.abs(point-1) < 1e-5, ~right_borders).any()
            for point in points
        ])
        connect_points_for_use = points[
            np.logical_or(left_connect_for_use, right_connect_for_use)
        ]
        return connect_points_for_use
        
    def global_solve(
        self,
        solver="np",
        calculate=True,
        verbose=False,
        alpha=0.0,
        **kwargs,
    ):
        A, b = self.generate_global_system(**kwargs)
        if calculate:
            res = self._solver(A, b, solver=solver, verbose=verbose, alpha=alpha)
            inds = [list(range(size)) for size in self.dim_sizes]
            all_cells = list(itertools.product(*inds))

            cell_shape = tuple([self.power] * self.n_dims)
            cell_size = np.prod(cell_shape)
            size = int(cell_size * self.n_funcs)

            for cell in all_cells:
                cell_index = self.cell_index(cell)
                cell_res = res[size * cell_index : size * (cell_index + 1)]

                for i in range(self.n_funcs):
                    self.cells_coefs[(i, *cell)] = cell_res[
                        i * cell_size : (i + 1) * cell_size
                    ].reshape(cell_shape)
        return A, b


# ______________________________TESTING________________________

if __name__ == "__main__":

    colloc_points = utils.f_collocation_points(5)

    power = 4
    function_list = ['u']
    variable_list = ['x']
    customs = {'small': 1e-5}
    params = {
        "n_dims": 1,
        "dim_sizes": np.array([3]),
        "area_lims": np.array([[0, 1]]),
        "power": power,
    }
    sol = Solution(**params)

    w = sol.steps[0] / 2
    def lp(lines_list, function_list=function_list, variable_list = variable_list, customs=customs):
        line_res = np.array([utils.lp(line, function_list=function_list, variable_list=variable_list, customs=customs) for line in lines_list]).T
        return line_res

    colloc_lines = ['(d/dx)^3 u = np.exp(x) * (x**4 + 14 * (x**3) + 49 * (x**2) + 32 * x - 12)']
    
    initial_string = '1' #'int(x[0] < sol.area_lims[0, 0] + small)'
    border_lines = [initial_string + '* u = ' + initial_string + ' * 0.']

    colloc_ops = lp(colloc_lines, customs=customs)
    border_ops = lp(border_lines, customs=customs)
    
    k1,k2 = 1,1
    func_num = 0
    connect_left_operators = [
        lambda __, _, u_bas, x, x_loc, func_num=func_num: k1
        * u_bas(0 * dir(x_loc), func_num)
        + k2 * np.sum(dir(x_loc)) * u_bas([0, 1], func_num),
        lambda __, _, u_bas, x, x_loc, func_num=func_num: k1
        * u_bas(2 * dir(x_loc), func_num)
        + k2 * np.sum(dir(x_loc)) * u_bas([0, 3], func_num)
    ]
    connect_right_operators = [
        lambda __, _, u_bas, x, x_loc, func_num=func_num: k1
        * u_bas(0 * dir(x_loc), func_num)
        - k2 * np.sum(dir(x_loc)) * u_bas([0, 1], func_num),
        lambda __, _, u_bas, x, x_loc, func_num=func_num: k1
        * u_bas(2 * dir(x_loc), func_num)
        - k2 * np.sum(dir(x_loc)) * u_bas([0, 3], func_num)
    ]
    connect_ops = [connect_left_operators, connect_right_operators]

    connect_points = np.array([[-1], [1]])
    border_points = connect_points
    
    points = (colloc_points, connect_points, border_points)

    iteration_dict = {
        "points": points,
        "colloc_ops": colloc_ops,
        "border_ops": border_ops,
        #"connect_ops": connect_ops,
    }
    
    A, b = sol.global_solve(calculate=False, **iteration_dict)
    print(A.shape)
    print(A)
    print(b)
    
    A, b = sol.global_solve(**iteration_dict)
    #for i in range(20):
    #    sol.iterate_cells(**iteration_dict)
    utils.plot(sol)
