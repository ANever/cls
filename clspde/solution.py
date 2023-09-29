import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
from math import comb
import re

from .basis import Basis
from .qr_solver import QR_solve, SVD_solve

def lp(line, function_list, variable_list):
    splited = line.split(' ')

    ops_stack = []

    def is_der_operator(string: str):
        if re.findall('\(d\/d..?\)', string):
            return True
        else:
            return False
        
    def apply_ops(ops_stack: list, func: str):
        dif_powers = [0]*len(variable_list)
        for op in ops_stack:
            op = op.replace('(d/d', '')
            op = op.replace(')', '')
            op = op.split('^')

            var_index = variable_list.index(op[0])
            try:
                power = op[1]
            except:
                power = 1
            dif_powers[var_index] = int(power)
        previous = ''
        if func[:2]=='&&':
            f_name = 'u_loc'
            previous = ',prev=True'
        elif func[:1]=='&':
            f_name = 'u_loc'
        else:
            f_name = 'u_bas'
        func_index = function_list.index(func.replace('&',''))
        return (f_name+'('+str(dif_powers)+', '+str(func_index)+ previous +')')

    def is_func(string:str):
        if string[:2]=='&&' and (string[2:] in function_list):
            return (True, 'prev')
        if string[:1]=='&' and (string[1:] in function_list):
            return (True, 'local')
        if string in function_list:
            return (True, 'basis')
        else:
            return (False, None)

    res = ''
    for i in range(len(splited)):
        if is_der_operator(splited[i]):
            ops_stack.append(splited[i])
        elif is_func(splited[i])[0]:
            res += (apply_ops(ops_stack, splited[i],))
            ops_stack = []
        else:
            res += splited[i]
    return res

def concat(a:np.ndarray, b:np.ndarray):
    a = np.array(a)
    b = np.array(b)
    if b.size == 0:
        return a
    if a.size == 0:
        return a
    else:
        return np.concatenate((a, b))

def f_collocation_points(N):
    points = np.zeros(N+1)
    h = 2/(N+1)
    points[0] = -1 + h/2
    for i in range(1, N+1):
        points[i] = points[i-1] + h
    return np.array(points).reshape(N+1,1)

class Solution():
    """ Solution of pde system

    Main class, that creates linear system of equations from pde and solves it

    Attributes:
        n_dims: amount of arguments of solution funciton
        dim_sizes: amount of cells in every direction (cell grid is square)
        area_lims: list of limitis of area in every dimention
        power: number of basis elements for every dimention
        basis: basis class to create basis elements
        n_funcs: amount of solution funtions

        cells_coefs: coefs of solution decomposition, must be addresed as [func_num, cell_addres, x1_power, x2_power..., xn_power]
    """

    def __init__(self, n_dims: int, dim_sizes: np.ndarray, area_lims: np.ndarray, power:int, basis: Basis, n_funcs:int = 1) -> None:
        self.area_lims = np.array(area_lims)
        self.n_dims = n_dims # = len(dim_sizes)
        self.dim_sizes = np.array(dim_sizes) # n of steps for all directions 
        self.power = power
        self.n_funcs = n_funcs
        self.init_grid()
        self.steps = ((self.area_lims[:,1] - self.area_lims[:,0]) / self.dim_sizes)
        
        self.basis = Basis(power, steps=self.steps, n_dims = n_dims)
        
        self.split_mats_inited = False
        self.precalculated_basis = False

    def init_grid(self) -> None:
        self.cells_shape = tuple([self.n_funcs] + list(self.dim_sizes) + [self.power]*self.n_dims)
        self.cells_coefs = np.zeros(self.cells_shape)
        self.prev_coefs = np.zeros(self.cells_shape)
        self.cell_size = self.n_funcs * (self.power**self.n_dims)

    def precalculate_basis(self, points: np.ndarray, max_der: np.ndarray) -> None:
        self.points = list(np.unique(points, axis=1))
        self.basis_evaled = np.empty((len(points), max_der + 1,max_der + 1, self.n_dims, self.power))
        self.basis_evaled_raveled = np.empty((len(points), max_der + 1,max_der + 1, self.power**self.n_dims))
        for point_num in range(len(points)):
            point = points[point_num]
            for der_t in range(max_der):
                for der_x in range(max_der):
                    self.basis_evaled[point_num, der_t,der_x] = self.basis.eval(point, np.array([der_t,der_x]), ravel=False)
                    self.basis_evaled_raveled[point_num, der_t,der_x] = np.outer(self.basis_evaled[point_num, der_t,der_x, 0],
                                                                                self.basis_evaled[point_num, der_t,der_x, 1]).ravel()
        self.precalculated_basis = True

    def point_num(self, point, threshold = 1e-3):
        """returns point addres in precalculated basis
        """
        return np.where(self.points==point)[0][0]
        
    def localize(self, global_point: np.ndarray, cells_closed_right: bool = False) -> np.ndarray:
        """returns cell addres and local coordinates in this cell of a globall coordinates
        """
        if cells_closed_right:
            shift = np.array(((global_point - self.area_lims[:,0]) % self.steps) < 1e-12, dtype=int)
            cell_num = np.array(np.floor((global_point - self.area_lims[:,0]) / self.steps) - shift, dtype=int)
        else:
            cell_num = np.array(np.floor((global_point - self.area_lims[:,0]) / self.steps), dtype=int)

        local_point = 2 * ((np.array(global_point)-self.area_lims[:,0])/np.array(self.steps) - (np.array(cell_num) + 0.5))

        return np.array([cell_num, local_point])

    def globalize(self, cell_num: np.ndarray, local_point: np.ndarray) -> np.ndarray:
        """returns globall coordinates of a point from its cell adress and local coordinates
        """
        global_point = self.area_lims[:,0] + ((np.array(local_point) + 1) + 2*np.array(cell_num)) * self.steps/2
        return global_point

    def init_split_mats(self):
        """creates matrices for cell division
        """
        self.minus_shift = np.zeros([self.power, self.power])
        self.plus_shift = np.zeros([self.power, self.power])
        for i in range(self.power):
            for j in range(i):
                self.minus_shift[i,j] = comb(i,j) * (-2)**i
        self.plus_shift = np.abs(self.minus_shift)

        self.split_mats_inited = True

    def cell_division(self, dimention:int = 0):
        """divides cells in half in certain dimention with saving of solution
        """
        plus_shift_mat = np.zeros((self.power,self.power))
        minus_shift_mat = np.zeros((self.power,self.power))
        for i in range(self.power):
            for j in range(i+1):
                plus_shift_mat[i,j] = comb(i,j)*2**(-(i))
                minus_shift_mat[i,j] = comb(i,j)*2**(-(i))*(-1)**(i+j)
        inds = [list(range(size)) for size in self.dim_sizes]
        old_cells_inds = list(itertools.product(*inds))
        old_cells_coefs = copy.deepcopy(self.cells_coefs)

        self.dim_sizes[dimention] *= 2
        self.steps = ((self.area_lims[:,1] - self.area_lims[:,0]) / self.dim_sizes)
        self.init_grid()
        cell_index_adder = np.zeros(self.n_dims, int)
        cell_index_adder[dimention] = 1
        
        for func in range(self.n_funcs):
            for cell in old_cells_inds:
                if dimention%2:
                    fst_cell = tuple(2**cell_index_adder[i] * cell[i] for i in range(self.n_dims))
                    self.cells_coefs[func][fst_cell] = np.tensordot(old_cells_coefs[func][cell], minus_shift_mat, axes=([dimention],[0]))
                    sec_cell = tuple(2**cell_index_adder[i] * cell[i] + cell_index_adder[i] for i in range(self.n_dims))
                    self.cells_coefs[func][sec_cell] = np.tensordot(old_cells_coefs[func][cell], plus_shift_mat, axes=([dimention],[0]))
                else: #transpose if needed
                    fst_cell = tuple(2**cell_index_adder[i] * cell[i] for i in range(self.n_dims))
                    self.cells_coefs[func][fst_cell] = np.tensordot(minus_shift_mat,old_cells_coefs[func][cell], axes=([0],[dimention]))
                    sec_cell = tuple(2**cell_index_adder[i] * cell[i] + cell_index_adder[i] for i in range(self.n_dims))
                    self.cells_coefs[func][sec_cell] = np.tensordot(plus_shift_mat, old_cells_coefs[func][cell], axes=([0],[dimention])) 

    def powerup(self):
        """raises power(amount of basis elements) of solution without change in it
        """
        rows=np.array(range(self.power))
        columns = rows

        self.power = self.power + 1
        old_cell_coefs = copy.deepcopy(self.cells_coefs)
        self.init_grid()

        self.Basis=Basis(self.power + 1, steps=self.steps, n_dims = self.n_dims)

        inds = [list(range(size)) for size in self.dim_sizes]
        old_cells_inds = list(itertools.product(*inds))

        for func in range(self.n_funcs):
            for cell in old_cells_inds:
                self.cells_coefs[func][cell][rows[:,np.newaxis],columns] = old_cell_coefs[func][cell]


    def eval(self, 
             point: np.ndarray, 
             derivatives: np.ndarray, 
             func:int = 0, 
             cell_num = None, 
             local = False, 
             cells_closed_right: bool = False, 
             coefs = None,
             prev = False):  #->float

        '''evaluation of solution function with argument x and list of partial derivatives
        
        Args:
            x: point to evaluate solution in, np.array(n_dim, float)
            derivatives: list of derivatives to take before evalution, np.array(n_dim, int)
            func: number of a solution function to evaluate
            local: whether x is local or not
            cell_num: cell addres of a local x
        '''
        derivatives = np.abs(derivatives)
        
        if prev:
            coefs = self.prev_coefs
        else:
            coefs = self.cells_coefs
        if local:
            local_point = point
        else:
            cell_num, local_point = self.localize(point, cells_closed_right)
        coefs = coefs[tuple(np.insert(np.array(cell_num, dtype=int), 0, func))]
        result = copy.deepcopy(coefs)
        #applying coefs tensor to evaled basis in point
        if self.precalculated_basis:
            try:
                point_num = self.point_num(local_point)
                basis_evaled = self.basis_evaled[point_num][tuple(derivatives)]
            except IndexError: #actually i forgot what error should be here
                pass
        else:
            basis_evaled = self.basis.eval(local_point, derivatives, ravel=False)
        for b_e in basis_evaled[::-1]:
            result = result.dot(b_e)
        return result

    def prepare_ops(self, list_of_ops, function_list, variable_list):
        for i in range(len(list_of_ops)):
            op = list_of_ops[i]
            if isinstance(op, str):
                list_of_ops[i] = lp(op, function_list, variable_list)

    def generate_system(self, cell_num: np.ndarray, points: np.ndarray, colloc_ops, border_ops, connect_ops = [],
                        function_list = ['u', 'v'], variable_list=['x','y']) -> tuple:
        colloc_points, connect_points, border_points = points

        def dir(point: np.ndarray) -> np.ndarray:
            direction = (np.abs(point) == 1) * (np.sign(point))
            return np.array(direction, dtype=int)

        w = (self.steps[0]/2) # weight
        # default connection
        if len(connect_ops) == 0:
            connect_left_operators = []
            connect_right_operators = []
            for func_num in range(self.n_funcs):
                # func_num=func_num is carrying that is needed to distinguish labmdas
                connect_left_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) + np.sum(dir(x_loc)) * u_bas(dir(x_loc),func_num) * w,
                                           lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(2*dir(x_loc),func_num)* w**2 + np.sum(dir(x_loc)) * u_bas(3*dir(x_loc),func_num)* w**3]
                connect_right_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) + np.sum(dir(x_loc))*u_bas(dir(x_loc),func_num)* w,
                                            lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(2*dir(x_loc),func_num) * w**2 + np.sum(dir(x_loc)) * u_bas(3*dir(x_loc),func_num)* w**3]

            connect_ops = [connect_left_operators, connect_right_operators]

        #default colloc points
        if len(colloc_points) == 0:
            colloc_points = f_collocation_points(self.power)
        
        connect_left_operators, connect_right_operators = connect_ops

        colloc_left_operators, colloc_right_operators = colloc_ops

        border_left_operators, border_right_operators = border_ops

        for ops in [connect_left_operators, connect_right_operators, colloc_left_operators, colloc_right_operators, border_left_operators, border_right_operators]:
            self.prepare_ops(ops, function_list, variable_list)

        colloc_mat, colloc_r = self.generate_subsystem(colloc_left_operators, colloc_right_operators, cell_num, colloc_points)
        
        left_borders = cell_num == np.zeros(self.n_dims)
        right_borders = cell_num == (self.dim_sizes-1)
        
        left_border_for_use = np.array([np.logical_and(point == -1, left_borders).any() for point in border_points])
        right_border_for_use = np.array([np.logical_and(point == 1, right_borders).any() for point in border_points])
        border_points_for_use = border_points[np.logical_or(left_border_for_use, right_border_for_use)]

        border_mat, border_r = self.generate_subsystem(border_left_operators, border_right_operators, cell_num, border_points_for_use)
        
        left_connect_for_use = np.array([np.logical_and(point == -1, ~left_borders).any() for point in connect_points])
        right_connect_for_use = np.array([np.logical_and(point == 1, ~right_borders).any() for point in connect_points])
        connect_points_for_use = connect_points[np.logical_or(left_connect_for_use, right_connect_for_use)] 
        
        connect_mat, connect_r = self.generate_subsystem(connect_left_operators, connect_right_operators, cell_num, connect_points_for_use)
        connect_weight = 1

        # print('normalized')
        def normalize(mat, r):
            coef = np.mean(mat[mat!=0])
            mat /= coef
            r /= coef
            return mat, r
        
        for (mat, r) in zip([colloc_mat, connect_mat, border_mat],[colloc_r, connect_r, border_r]):
            mat ,r = normalize(mat, r)

        res_mat = concat(concat(colloc_mat, border_mat), connect_mat * connect_weight)
        res_right = concat(concat(colloc_r, border_r), connect_r * connect_weight)

        return res_mat, res_right

    def iterate_cells(self, solver='np', **kwargs) -> None:
        """solves local system of equations for every cell once
        """
        inds = [list(range(size)) for size in self.dim_sizes]
        all_cells = list(itertools.product(*inds))
        cell_shape = tuple([self.power]*self.n_dims)
        for cell in all_cells:
            mat, right = self.generate_system(cell, **kwargs)
            cell_size = np.prod(cell_shape)
            solution = self._solver(A = mat, b = right, solver=solver)
            for i in range(self.n_funcs):
                self.cells_coefs[(i,*cell)] = solution[i*cell_size:(i+1)*cell_size].reshape(cell_shape)

    def _solver(self, A, b, solver = 'np', svd_threshold = 1e-5, verbose = False):
        """solves system Ax=b with chosen solver"""
        if solver == 'QR':
            res = QR_solve(A,b)
        elif solver == 'np':
            b = np.transpose(A) @ b
            A = np.transpose(A) @ A
            res = np.linalg.solve(A,b)
        elif solver == 'SVD':
            res = SVD_solve(A,b, threshold=svd_threshold, verbose = verbose)
        else: 
            raise np.ERR_DEFAULT
        return res

    def solve(self, threshold = 1e-5, max_iter = 10000, verbose=False, **kwargs) -> None:
        """solves set problem with iterative solution of local systems in every cell 
        untill convergence
        """
        prev_coefs = copy.deepcopy(self.cells_coefs)
        for i in range(max_iter):
            self.iterate_cells(**kwargs)
            residual = np.max(np.abs((prev_coefs - self.cells_coefs)))
            if verbose:
                print(residual)
            if residual < threshold:
                break
            prev_coefs = copy.deepcopy(self.cells_coefs)
        if verbose:
            print('Iterations to converge: ', i)

    def generate_integral(self, time):
        """TODO not implemented yet, for integral conditions
        """
        #generate common line
        n = self.power
        integral_cell = self.basis.eval([time, 1],[0,0],raver=True)#1/np.array(range(1,n+1)) * ([2, 0]*int(np.ceil(n/2)))[:n]
        full_line = np.zeros(np.prod(self.cells_shape))
        # a = np.zeros((self.power, self.power))
        # a[0] = integral_cell
        # cell_line = concat(np.ravel(a), np.ravel(a) * 0)

        inds = [list(range(size)) for size in self.dim_sizes]
        all_cells = list(itertools.product(*inds))
        num_of_vars = self.cell_size #will work only for 2d
        for cell_num in all_cells:
            if cell_num[0] == time:
                cell_ind = cell_num[1] + cell_num[0] * self.dim_sizes[1]
                full_line[cell_ind * num_of_vars:(cell_ind+1) * num_of_vars] = integral_cell

        return full_line, 1
        #TODO
        #set time moment
        #iterate over space
        #set common line into cells

    def generate_eq(self, cell_num, left_side_operator, right_side_operator, points):
        '''basic func for generating equation from leftside and rightside operators
        '''
        def left_side(operator, cell_num, point: np.ndarray) -> np.ndarray:
            '''must return row of coeficient for LSE'''
            loc_point = copy.deepcopy(point)
            global_point = self.globalize(cell_num, point)
            x = copy.deepcopy(global_point)
            def u_bas(der,func=0):
                bas_size = int(self.cell_size/self.n_funcs)
                result = np.zeros(self.n_funcs * bas_size)
                if self.precalculated_basis:
                    loc_point_num = self.point_num(loc_point)
                    result[func*bas_size:(func+1)*bas_size] = self.basis_evaled_raveled[loc_point_num, der[0],der[1]]
                else:
                    result[func*bas_size:(func+1)*bas_size] = self.basis.eval(loc_point, der, ravel=True)
                return result

            def u_loc(der,func=0,prev=False):
                try:
                    result = self.eval(loc_point, der, func=func, local = True, cell_num = cell_num, prev=prev)
                except IndexError:
                    result = self.eval(loc_point, der, func=func, local = True, cell_num = cell_num, prev=prev, cells_closed_right=True)
                return result

            return operator(u_loc, u_bas, x, loc_point)

        def right_side(operator, cell_num, point: np.ndarray) -> float:
            
            def dir(point: np.ndarray) -> np.ndarray:
                direction = (np.abs(point) == 1) * (np.sign(point)) 
                return direction

            global_point = self.globalize(cell_num, point)
            # x = global_point
            loc_point = copy.deepcopy(point)
            u_loc = lambda der, func_num=0, prev=False: self.eval(loc_point, der, local = True, cell_num = cell_num, func=func_num)   # for linearization purpses

            neigh_point = loc_point-2*dir(loc_point)

            u_nei = lambda der, func_num=0: self.eval(neigh_point, der, local = True, cell_num = cell_num + dir(loc_point), func=func_num)
            return operator(u_loc, u_nei, global_point, loc_point) #x
        
        mat = np.zeros((len(points), self.cell_size))
        r_side = np.zeros((len(points)))
        for i in range(len(points)):
            mat[i] = left_side(left_side_operator, cell_num,  points[i])
            r_side[i] = right_side(right_side_operator, cell_num, points[i])
        return mat, r_side

    def generate_subsystem(self, left_ops, right_ops, cell_num, points: np.ndarray) -> tuple:
        mat, r = self.generate_eq(cell_num, left_ops[0], right_ops[0], points)
        for i in range(1,len(left_ops)):
            mat_small, r_small = self.generate_eq(cell_num, left_ops[i], right_ops[i], points)
            mat = concat(mat, mat_small)
            r = concat(r, r_small)
        return mat, r

    def generate_connection_couple(self, left_ops, cell_num, points: np.ndarray) -> tuple:
        # left ops must be a pair of functions
        # right ops substitude
        right_ops = [lambda u, _, x, x_loc: 0] * len(left_ops[0])
        def dir(point: np.ndarray) -> np.ndarray:
            direction = (np.abs(point) == 1) * (np.sign(point))
            return np.array(direction, dtype=int)

        first = True
        for point in points:
            first_line, _ = self.generate_subsystem(left_ops[0], right_ops, cell_num, np.array([point]))
            neigh = tuple(np.array(cell_num) + dir(point))
            neigh_point = point - 2*dir(point)
            second_line, _ = self.generate_subsystem(left_ops[1], right_ops, neigh, np.array([neigh_point]))

            connect_line = np.zeros((len(left_ops[0]), np.prod(self.cells_coefs.shape)))

            index = self.cell_index(cell_num)
            neigh_index = self.cell_index(neigh)

            connect_line[:, index*self.cell_size:(index+1)*self.cell_size] = first_line
            connect_line[:, neigh_index*self.cell_size:(neigh_index+1)*self.cell_size] = -second_line
            if first:
                connect_mat = connect_line
                first = False
            else:
                connect_mat = concat(connect_mat, connect_line)
            # connect_mat.append(connect_line) #???
        return np.array(connect_mat)

    def plot(self, n = 100):
        func = np.zeros(n)
        grid = np.linspace(self.area_lims[0,0], self.area_lims[0,1], n, endpoint=False)
        for i in range(len(grid)): 
            func[i] = self.eval(grid[i], [0])
        plt.plot(func)
        plt.show()
    
    def plot2d(self, n=100, x_lims = None, y_lims = None, func_num=0, derivatives = [0,0], label = ['t','x'], func_name='', **plot_kwargs):
        func = np.zeros((n,n))
        if x_lims == None:
            x_lims = self.area_lims[0]
        if y_lims == None:
            y_lims = self.area_lims[1]
        ax1 = np.linspace(x_lims[0], x_lims[1], n, endpoint=False)
        ax2 = np.linspace(y_lims[0], y_lims[1], n, endpoint=False)
        X, Y = np.meshgrid(ax1, ax2)

        for i in range(n):
            for j in range(n): 
                func[j, i] = self.eval([ax1[i], ax2[j]], derivatives, func=func_num)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize = (7,7))
        surf = ax.plot_surface(X, Y, func, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, **plot_kwargs)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_title(func_name)
        # plt.show()


    def cell_index(self, cell_num):
        if self.n_dims == 2:
            cell_ind = cell_num[0] + cell_num[1] * self.dim_sizes[0] # + cell_num[2] * prod(self.dim_sizes[:2]...
        elif self.n_dims == 1:
            cell_ind = cell_num[0]
        else:
            raise LookupError
        return cell_ind

    def generate_global_system(self, points: np.ndarray, colloc_ops, border_ops, connect_ops = [], weights = [1,1,1], 
                               function_list = ['u', 'v'], variable_list=['x','y']) -> tuple:
         
        colloc_points, connect_points, border_points = points

        def dir(point: np.ndarray) -> np.ndarray:
            direction = (np.abs(point) == 1) * (np.sign(point))
            return np.array(direction, dtype=int)

        w = (self.steps[0]/2)#weight
        #default connection
        if len(connect_ops) == 0:
            connect_left_operators = []
            connect_right_operators = []
            for func_num in range(self.n_funcs):
                #func_num=func_num is carrying that is needed to distinguish lambdas
                
                connect_left_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) + np.sum(dir(x_loc)) * u_bas(dir(x_loc),func_num) * w,
                                            lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(2*dir(x_loc),func_num)* w**2 + np.sum(dir(x_loc)) * u_bas(3*dir(x_loc),func_num)* w**3]
                connect_right_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) - np.sum(dir(x_loc))*u_bas(dir(x_loc),func_num)* w,
                                            lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(2*dir(x_loc),func_num) * w**2 - np.sum(dir(x_loc)) * u_bas(3*dir(x_loc),func_num)* w**3]
            connect_ops = [connect_left_operators, connect_right_operators]

        #default colloc points
        if len(colloc_points) == 0:
            colloc_points = f_collocation_points(self.power)
        
        connect_left_operators, connect_right_operators = connect_ops

        num_of_vars = self.cell_size # np.prod(self.cells_coefs.shape)

        inds = [list(range(size)) for size in self.dim_sizes]
        all_cells = list(itertools.product(*inds))
        
        num_of_cells = len(all_cells)

        num_of_collocs = len(colloc_points) * len(colloc_ops[0])
        num_of_eqs = len(all_cells) * num_of_collocs

        global_colloc_mat = np.zeros((num_of_eqs, num_of_vars * num_of_cells))
        global_colloc_right = np.zeros(num_of_eqs)

        num_of_border = len(border_points)* len(border_ops[0])
        num_of_eqs = len(all_cells)*num_of_border

        global_border_mat = np.zeros((num_of_eqs, num_of_vars * num_of_cells))
        global_border_right = np.zeros(num_of_eqs)
        
        num_of_connect = len(connect_points)* len(connect_ops)
        num_of_eqs = len(all_cells)*num_of_connect
        
        global_connect_mat = []
        global_connect_right = np.zeros(num_of_eqs)
        
        colloc_left_operators, colloc_right_operators = colloc_ops
        border_left_operators, border_right_operators = border_ops

        for ops in [connect_left_operators, connect_right_operators, colloc_left_operators, colloc_right_operators, border_left_operators, border_right_operators]:
            self.prepare_ops(ops, function_list, variable_list)

        first_connect = True
        for cell_num in all_cells:
            left_borders = cell_num == np.zeros(self.n_dims)
            right_borders = cell_num == (self.dim_sizes-1)
            left_border_for_use = np.array([np.logical_and(point == -1, left_borders).any() for point in border_points])
            
            right_border_for_use = np.array([np.logical_and(point == 1, right_borders).any() for point in border_points])
            border_points_for_use = border_points[np.logical_or(left_border_for_use, right_border_for_use)]

            colloc_mat, colloc_r = self.generate_subsystem(colloc_left_operators, colloc_right_operators, cell_num, colloc_points)
            border_mat, border_r = self.generate_subsystem(border_left_operators, border_right_operators, cell_num, border_points_for_use)

            #for 2d only!
            # cell_ind = cell_num[0] + cell_num[1] * self.dim_sizes[0] # + cell_num[2] * prod(self.dim_sizes[:2]...
            cell_ind = self.cell_index(cell_num)
            global_colloc_mat[cell_ind * num_of_collocs:(cell_ind+1) * num_of_collocs, cell_ind * num_of_vars:(cell_ind+1) * num_of_vars] = colloc_mat
            global_colloc_right[cell_ind * num_of_collocs:(cell_ind+1) * num_of_collocs] = colloc_r
            
            num_of_border_to_use = border_mat.shape[0]
            global_border_mat[cell_ind * num_of_border:cell_ind * num_of_border + num_of_border_to_use, cell_ind * num_of_vars:(cell_ind+1) * num_of_vars] = border_mat
            global_border_right[cell_ind * num_of_border:cell_ind * num_of_border + num_of_border_to_use] = border_r

            left_connect_for_use = np.array([np.logical_and(point == -1, ~left_borders).any() for point in connect_points])
            right_connect_for_use = np.array([np.logical_and(point == 1, ~right_borders).any() for point in connect_points])
            connect_points_for_use = connect_points[np.logical_or(left_connect_for_use, right_connect_for_use)]
            
            connect_mat = self.generate_connection_couple([connect_left_operators, connect_right_operators], cell_num, connect_points_for_use)
            if first_connect:
                global_connect_mat = connect_mat
                first_connect = False
            else:
                global_connect_mat = concat(global_connect_mat, connect_mat)
        global_connect_mat = np.array(global_connect_mat)
        global_connect_right = np.zeros(len(global_connect_mat))

        # connect_w, border_w, connect_w = weights

        def normalize(mat, r, w):
            coef = np.max(np.abs(mat[mat!=0]))
            mat /= coef * w
            r /= coef * w
            return mat, r
        
        for (mat, r, w) in zip([global_colloc_mat, global_connect_mat, global_border_mat],[global_colloc_right, global_connect_right, global_border_right], weights):
            mat ,r = normalize(mat, r, w)

        res_mat = concat(concat(global_colloc_mat, global_border_mat), global_connect_mat)
        res_right = concat(concat(global_colloc_right, global_border_right), global_connect_right)

        notnull_ind = np.sum(res_mat != 0, axis=1)!=0
        res_mat = res_mat[notnull_ind]
        res_right = res_right[notnull_ind]

        return res_mat, res_right

    def global_solve(self, solver = 'np',return_system = False, calculate = True, svd_threshold = 1e-4, verbose = False, alpha=0,  **kwargs):
        A, b = self.generate_global_system(**kwargs)
        if alpha > 0:
            A = concat(A, np.eye(A.shape[1])*alpha)
            b = concat(b, np.zeros(A.shape[1]))
        if calculate:
            res = self._solver(A,b,solver=solver, svd_threshold=svd_threshold, verbose = verbose)
            inds = [list(range(size)) for size in self.dim_sizes]
            all_cells = list(itertools.product(*inds))

            cell_shape = tuple([self.power]*self.n_dims)
            cell_size = np.prod(cell_shape)
            size = int(cell_size*self.n_funcs)

            for cell in all_cells:
                cell_index = self.cell_index(cell)
                cell_res = res[size*cell_index:size*(cell_index+1)]

                for i in range(self.n_funcs):
                    self.cells_coefs[(i,*cell)] = cell_res[i*cell_size:(i+1)*cell_size].reshape(cell_shape)
        
        self.prev_coefs = copy.deepcopy(self.cells_coefs)
        if return_system:
            return A, b
    

#______________________________TESTING________________________

if __name__ == '__main__':
    
    def f_collocation_points(N):
        points = np.zeros(N+1)
        h = 2/(N+1)
        points[0] = -1 + h/2
        for i in range(1, N+1):
            points[i] = points[i-1] + h
        return np.array(points).reshape(N+1,1)
    colloc_points = f_collocation_points(4)


    power = 5
    params = {
        'n_dims': 1,
        'dim_sizes': np.array([5]),
        'area_lims': np.array([[0,1]]),
        'power': power,
        'basis': Basis(power),
    }
    sol = Solution(**params)

    w = (sol.steps[0]/2)

    colloc_left_operators = [lambda u_loc, u_bas, x, x_loc: u_bas([4]) * (w**4)]
    colloc_right_operators = [lambda u_loc, u_nei, x, x_loc: np.exp(x)*(x**4 + 14*(x**3) + 49*(x**2) + 32*x - 12) * (w**4)]
    colloc_ops = [colloc_left_operators, colloc_right_operators]

    border_left_operators = [lambda _, u_bas, x, x_loc: u_bas([0]), 
                                lambda _, u_bas, x, x_loc: u_bas([1]) * w]
    border_right_operators = [lambda u, _, x, x_loc: 0,
                                lambda u, _, x, x_loc: 0 * w]
    border_ops = [border_left_operators, border_right_operators]

    connect_points = np.array([[-1], [1]])
    border_points = connect_points

    points = (colloc_points, connect_points, border_points)

    iteration_dict = {'points':points,
                    'colloc_ops':colloc_ops,
                    'border_ops':border_ops}

    sol.iterate_cells(**iteration_dict)

    print('executed')