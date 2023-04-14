from re import I
import numpy as np
import copy
import itertools
from .basis import Basis

from .qr_solver import QR_solve, SVD_solve

import matplotlib.pyplot as plt
from matplotlib import cm

from math import comb

def concat(a:np.array, b:np.array):
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


# def comb(N,k): # from scipy.comb(), but MODIFIED!
#     if (k > N) or (N < 0) or (k < 0):
#         return 0
#     N,k = map(long,(N,k))
#     top = N
#     val = 1
#     while (top > (N-k)):
#         val *= top
#         top -= 1
#     n = 1
#     while (n < k+1):
#         val /= n
#         n += 1
#     return val

class Solution():

    def __init__(self, n_dims: int, dim_sizes: np.array, area_lims: np.array, power:int, basis: Basis, n_funcs:int = 1) -> None:
        '''
        initiation of solution
        init of grid of cells, initial coefs, basis
        area_lims - np.array of shape (n_dims, 2) - pairs of upper and lower limits of corresponding dim
        dim_sizes - np. array of shape (n_dims)
        '''

        self.area_lims = np.array(area_lims)
        self.n_dims = n_dims # = len(dim_sizes)
        self.dim_sizes = np.array(dim_sizes) # n of steps for all directions 
        self.power = power
        self.n_funcs = n_funcs
        self.init_grid()
        self.steps = ((self.area_lims[:,1] - self.area_lims[:,0]) / self.dim_sizes)
        
        self.basis = Basis(power, steps=self.steps, n_dims = n_dims)
        
        self.split_mats_inited = False

    def init_grid(self) -> None:
        self.cells_shape = tuple([self.n_funcs] + list(self.dim_sizes) + [self.power]*self.n_dims)
        self.cells_coefs = np.ones(self.cells_shape) * 0.4
        self.cell_size = self.n_funcs * (self.power**self.n_dims)
        
    def localize(self, global_point: np.array, cells_closed_right: bool = False) -> np.array:
        if cells_closed_right:
            shift = np.array(((global_point - self.area_lims[:,0]) % self.steps) < 1e-12, dtype=int)
            cell_num = np.array(np.floor((global_point - self.area_lims[:,0]) / self.steps) - shift, dtype=int)
        else:
            cell_num = np.array(np.floor((global_point - self.area_lims[:,0]) / self.steps), dtype=int)

        local_point = 2 * ((np.array(global_point)-self.area_lims[:,0])/np.array(self.steps) - (np.array(cell_num) + 0.5))

        return cell_num, local_point

    def globalize(self, cell_num: np.array, local_point: np.array) -> np.array:
        global_point = self.area_lims[:,0] + (np.array(local_point) + 2*np.array(cell_num) + 1) * self.steps/2
        return global_point

    def init_split_mats(self):
        self.minus_shift = np.zeros([self.power, self.power])
        self.plus_shift = np.zeros([self.power, self.power])
        for i in range(self.power):
            for j in range(i):
                self.minus_shift[i,j] = comb(i,j) * (-2)**i
        self.plus_shift = np.abs(self.minus_shift)

        self.split_mats_inited = True




    #only for 2d use
    def split_cells(self):
        if not(self.split_mats_inited):
            self.init_split_mats()
        old_coefs = copy.deepcopy(self.cells_coefs)
        inds = [list(range(size)) for size in self.dim_sizes]
        all_old_cells = list(itertools.product(*inds))
        
        self.dim_sizes = self.dim_sizes*2

        self.self.init_grid()
        self.steps = ((self.area_lims[:,1] - self.area_lims[:,0]) / self.dim_sizes)
        
        for cell in all_old_cells:
            sol.cells_coefs[tuple(cell+np.array([0,0]))] = np.transpose(self.minus_shift) @ old_coefs[cell] @ self.minus_shift
            sol.cells_coefs[tuple(cell+np.array([1,0]))] = np.transpose(self.plus_shift) @ old_coefs[cell] @ self.minus_shift
            sol.cells_coefs[tuple(cell+np.array([0,1]))] = np.transpose(self.minus_shift) @ old_coefs[cell] @ self.plus_shift
            sol.cells_coefs[tuple(cell+np.array([1,1]))] = np.transpose(self.plus_shift) @ old_coefs[cell] @ self.plus_shift

    def eval(self, point: np.array, derivatives: np.array, func:int = 0, cell_num = None, local = False, cells_closed_right: bool = False) -> float:
        '''
        x - np.array(n_dim, float)
        derivatives - np.array(n_dim, int)
        evaluation of solution function with argument x and list of partial derivatives
        '''
        derivatives = np.abs(derivatives)
        
        if local:
            local_point = point
        else:
            cell_num, local_point = self.localize(point, cells_closed_right)
        coefs = self.cells_coefs[tuple(np.insert(np.array(cell_num, dtype=int), 0, func))]
        result = copy.deepcopy(coefs)
        #applying coefs tensor to evaled basis in point
        basis_evaled = self.basis.eval(local_point, derivatives, ravel=False)
        for b_e in basis_evaled[::-1]:
            result = result.dot(b_e)
        return result

    def generate_system(self, cell_num: np.array, points: np.array, colloc_ops, border_ops, connect_ops = []) -> tuple:
        colloc_points, connect_points, border_points = points

        def dir(point: np.array) -> np.array:
            direction = (np.abs(point) == 1) * (np.sign(point))
            return np.array(direction, dtype=int)

        w = (self.steps[0]/2) # weight
        # default connection
        if len(connect_ops) == 0:
            connect_left_operators = []
            connect_right_operators = []
            for func_num in range(self.n_funcs):
                # func_num=func_num is carrying that is needed to distinguish labmdas
                '''
                strong operators for diffrential eqs with 4th derivative
                # connect_left_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) + np.sum(dir(x_loc)) * u_bas(dir(x_loc),func_num) * w,
                #                     lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(2*dir(x_loc),func_num)* w**2 + np.sum(dir(x_loc)) * u_bas(3*dir(x_loc),func_num)* w**3]
                # connect_right_operators += [lambda _, u_nei, x, x_loc, func_num=func_num: u_nei(0*dir(x_loc),func_num) + np.sum(dir(x_loc))*u_nei(dir(x_loc),func_num)* w,
                #                             lambda _, u_nei, x, x_loc, func_num=func_num: u_nei(2*dir(x_loc),func_num) * w**2 + np.sum(dir(x_loc)) * u_nei(3*dir(x_loc),func_num)* w**3]
                '''
                # connect_left_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) + np.sum(dir(x_loc)) * u_bas(dir(x_loc),func_num) * w,]
                # connect_right_operators += [lambda _, u_nei, x, x_loc, func_num=func_num: u_nei(0*dir(x_loc),func_num) + np.sum(dir(x_loc))*u_nei(dir(x_loc),func_num)* w,]
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
        res_mat = concat(concat(colloc_mat, border_mat), connect_mat * connect_weight)
        res_right = concat(concat(colloc_r, border_r), connect_r * connect_weight)

        return res_mat, res_right

    def iterate_cells(self, **kwargs) -> None:
        inds = [list(range(size)) for size in self.dim_sizes]
        all_cells = list(itertools.product(*inds))
        cell_shape = tuple([self.power]*self.n_dims)
        for cell in all_cells:
            mat, right = self.generate_system(cell, **kwargs)
            cell_size = np.prod(cell_shape)
            solution = self._solver(A = mat, b = right, solver='np')
            for i in range(self.n_funcs):
                self.cells_coefs[(i,*cell)] = solution[i*cell_size:(i+1)*cell_size].reshape(cell_shape)


    def _solver(self, A, b, solver = 'np', svd_threshold = 1e-5, verbose = False):
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
        #generate common line
        n = self.power
        integral_cell = 1/np.array(range(1,n+1)) * ([2, 0]*int(np.ceil(n/2)))[:n]
        full_line = np.zeros(np.prod(self.cells_shape))
        a = np.zeros((self.power, self.power))
        a[0] = integral_cell
        cell_line = concat(np.ravel(a), np.ravel(a) * 0)

        inds = [list(range(size)) for size in self.dim_sizes]
        all_cells = list(itertools.product(*inds))
        num_of_vars = self.cell_size #will work only for 2d
        for cell_num in all_cells:
            if cell_num[0] == time:
                cell_ind = cell_num[1] + cell_num[0] * self.dim_sizes[1]
                full_line[cell_ind * num_of_vars:(cell_ind+1) * num_of_vars] = cell_line
        return full_line
        #set time moment
        #iterate over space
        #set common line into cells


    def generate_eq(self, cell_num, left_side_operator, right_side_operator, points):
        '''
        basic func for generating equation
        '''
        def left_side(operator, cell_num, point: np.array) -> np.array:
            '''must return row of coeficient for LSE'''
            loc_point = copy.deepcopy(point)
            global_point = self.globalize(cell_num, point)
            x = copy.deepcopy(global_point)
            def u_bas(der,func=0):
                bas_size = int(self.cell_size/self.n_funcs)
                result = np.zeros(self.n_funcs * bas_size)
                result[func*bas_size:(func+1)*bas_size] = self.basis.eval(loc_point, der, ravel=True)
                return result

            def u_loc(der,func=0):
                try:
                    result = self.eval(loc_point, der, func=func, local = True, cell_num = cell_num)
                except:
                    result = self.eval(loc_point, der, func=func, local = True, cell_num = cell_num, cells_closed_right=True)
                return result

            return operator(u_loc, u_bas, x, loc_point)

        def right_side(operator, cell_num, point: np.array) -> float:
            
            def dir(point: np.array) -> np.array:
                direction = (np.abs(point) == 1) * (np.sign(point)) 
                return direction

            global_point = self.globalize(cell_num, point)
            x = global_point
            loc_point = copy.deepcopy(point)            
            u_loc = lambda der, func_num=0: self.eval(loc_point, der, local = True, cell_num = cell_num, func=func_num)   # for linearization purpses

            neigh_point = loc_point-2*dir(loc_point)

            u_nei = lambda der, func_num=0: self.eval(neigh_point, der, local = True, cell_num = cell_num + dir(loc_point), func=func_num)
            return operator(u_loc, u_nei, global_point, loc_point) #x
        
        mat = np.zeros((len(points), self.cell_size))
        r_side = np.zeros((len(points)))
        for i in range(len(points)):
            mat[i] = left_side(left_side_operator, cell_num,  points[i])
            r_side[i] = right_side(right_side_operator, cell_num, points[i])
        return mat, r_side

    def generate_subsystem(self, left_ops, right_ops, cell_num, points: np.array) -> tuple:
        mat, r = self.generate_eq(cell_num, left_ops[0], right_ops[0], points)
        for i in range(1,len(left_ops)):
            mat_small, r_small = self.generate_eq(cell_num, left_ops[i], right_ops[i], points)
            mat = concat(mat, mat_small)
            r = concat(r, r_small)
        return mat, r

    def generate_connection_couple(self, left_ops, cell_num, points: np.array) -> tuple:
        # left ops must be a pair of functions
        # right ops substitude
        right_ops = [lambda u, _, x, x_loc: 0] * len(left_ops[0])              
        
        # print(cell_num)

        def dir(point: np.array) -> np.array:
            direction = (np.abs(point) == 1) * (np.sign(point))
            return np.array(direction, dtype=int)

        func_num = 0 #IS IT OK???

        # connect_mat = []
        
        first = True
        for point in points:
            first_line, _ = self.generate_subsystem(left_ops[0], right_ops, cell_num, np.array([point]))
            # print(cell_num, point, dir(point))
            neigh = tuple(np.array(cell_num) + dir(point))
            neigh_point = point - 2*dir(point)
            second_line, _ = self.generate_subsystem(left_ops[1], right_ops, neigh, np.array([neigh_point]))

            connect_line = np.zeros((len(left_ops[0]), np.prod(self.cells_coefs.shape)))

            index = self.cell_index(cell_num) #cell_num[1] + cell_num[0] * self.dim_sizes[1] #np.prod(self.dim_sizes[0])
            # neigh_index = neigh[0] + neigh[1] * np.prod(self.dim_sizes[:1])
            neigh_index = self.cell_index(neigh) #neigh[1] + neigh[0] * self.dim_sizes[1] #np.prod(self.dim_sizes[1])
            
            connect_line[:, index*self.cell_size:(index+1)*self.cell_size] = first_line
            connect_line[:, neigh_index*self.cell_size:(neigh_index+1)*self.cell_size] = -second_line

            # print('neigh and point', neigh, point, neigh_point)
            # print(connect_line)
            
            #connect_line = np.array(connect_line)

            if first:
                connect_mat = connect_line
                first = False
            else:
                connect_mat = concat(connect_mat, connect_line)
            # connect_mat.append(connect_line) #???
        return np.array(connect_mat)

    # def cell_division(self, dimention:int = 0):
    #     plus_shift_mat = np.zeros((self.power,self.power))
    #     minus_shift_mat = np.zeros((self.power,self.power))
    #     for i in range(self.power):
    #         for j in range(i):
    #             plus_shift_mat[i,j] = comb(i,j)*2**(-(i+j))*(-1)**j
    #             minus_shift_mat[i,j] = comb(i,j)*2**(-(i+j))
        
    #     inds = [list(range(size)) for size in self.dim_sizes]
    #     old_cells = list(itertools.product(*inds))

    #     self.dim_sizes[0] *= 2
    #     self.steps = ((self.area_lims[:,1] - self.area_lims[:,0]) / self.dim_sizes)
        
    #     self.cells_coefs = np.ones(self.cells_shape) 
        
    #     for func in range(self.n_funcs):
    #         for cell in old_cells:
    #             new_cell_coefs[func, cell] = self.cell_coefs
    #         for i in range(self.dim_sizes[dimention]):
    #             pass 
    #         #TODO FINISH

    #     self.cells_shape = tuple([self.n_funcs] + list(self.dim_sizes) + [self.power]*self.n_dims)
    #     self.cells_coefs = new_cell_coefs
    #     self.cell_size = self.n_funcs * (self.power**self.n_dims)

    def plot(self, n = 100):
        func = np.zeros(n)
        grid = np.linspace(self.area_lims[0,0], self.area_lims[0,1], n, endpoint=False)
        for i in range(len(grid)): 
            func[i] = self.eval(grid[i], [0])
        plt.plot(func)
        plt.show()
    
    def plot2d(self, n=100, x_lims = None, y_lims = None, func_num=0, derivatives = [0,0]):
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
                        linewidth=0, antialiased=False)

        # ax.set_xticks(X)
        # ax.set_xticks(Y)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        plt.show()


    def cell_index(self, cell_num):
        if self.n_dims == 2:
                cell_ind = cell_num[1] + cell_num[0] * self.dim_sizes[1] # + cell_num[2] * prod(self.dim_sizes[:2]...
        elif self.n_dims == 1:
            cell_ind = cell_num[0]
        else:
            raise LookupError
        return cell_ind

    def generate_global_system(self, points: np.array, colloc_ops, border_ops, connect_ops = [], connect_weight=1) -> tuple:

        colloc_points, connect_points, border_points = points

        def dir(point: np.array) -> np.array:
            direction = (np.abs(point) == 1) * (np.sign(point))
            return np.array(direction, dtype=int)

        w = (self.steps[0]/2)#weight
        #default connection
        if len(connect_ops) == 0:
            connect_left_operators = []
            connect_right_operators = []
            for func_num in range(self.n_funcs):
                #func_num=func_num is carrying that is needed to distinguish labmdas
                
                connect_left_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) + np.sum(dir(x_loc)) * u_bas(dir(x_loc),func_num) * w,
                                           lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(2*dir(x_loc),func_num)* w**2 + np.sum(dir(x_loc)) * u_bas(3*dir(x_loc),func_num)* w**3]
                connect_right_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) - np.sum(dir(x_loc))*u_bas(dir(x_loc),func_num)* w,
                                            lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(2*dir(x_loc),func_num) * w**2 - np.sum(dir(x_loc)) * u_bas(3*dir(x_loc),func_num)* w**3]
                '''
                strong operators for equations with 4th derivative 
                '''
                # connect_left_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) + np.sum(dir(x_loc)) * u_bas(dir(x_loc),func_num) * w,]
                # connect_right_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) - np.sum(dir(x_loc))*u_bas(dir(x_loc),func_num)* w,]
                
                # loc_point = [1,0.5]
                # def u_bas(der,func=0):
                #     bas_size = int(self.cell_size/self.n_funcs)
                #     result = np.zeros(self.n_funcs * bas_size)
                #     result[func*bas_size:(func+1)*bas_size] = self.basis.eval(loc_point, der, ravel=True)
                #     return result
            
            # connect_left_operators = [lambda _, u_bas, x, x_loc: u_bas(0*dir(x_loc),func_num) + np.sum(dir(x_loc)) * u_bas(dir(x_loc),func_num) * w for func_num in range(self.n_funcs)]+\
            #                         [lambda _, u_bas, x, x_loc: u_bas(2*dir(x_loc),func_num)* w**2 + np.sum(dir(x_loc)) * u_bas(3*dir(x_loc),func_num)* w**3 for func_num in range(self.n_funcs)]
            # connect_right_operators += [lambda _, u_nei, x, x_loc: u_nei(0*dir(x_loc),func_num) + np.sum(dir(x_loc))*u_nei(dir(x_loc),func_num)* w for func_num in range(self.n_funcs)]+\
            #                                 [lambda _, u_nei, x, x_loc: u_nei(2*dir(x_loc),func_num) * w**2 + np.sum(dir(x_loc)) * u_nei(3*dir(x_loc),func_num)* w**3 for func_num in range(self.n_funcs)]

            connect_ops = [connect_left_operators, connect_right_operators]

        #default colloc points
        if len(colloc_points) == 0:
            colloc_points = f_collocation_points(self.power)
        
        connect_left_operators, connect_right_operators = connect_ops

        num_of_vars = self.cell_size # np.prod(self.cells_coefs.shape)

        inds = [list(range(size)) for size in self.dim_sizes]
        all_cells = list(itertools.product(*inds))
        cell_shape = tuple([self.power]*self.n_dims)

        num_of_collocs = len(colloc_points) * len(colloc_ops[0])
        num_of_eqs = len(all_cells) * num_of_collocs
        num_of_cells = len(all_cells)

        global_colloc_mat = np.zeros((num_of_eqs, num_of_vars * num_of_cells))
        global_colloc_right = np.zeros(num_of_eqs)

        num_of_border = len(border_points)* len(border_ops[0])
        num_of_eqs = len(all_cells)*num_of_border

        # global_border_mat = np.zeros((num_of_eqs, len(all_cells) * np.prod(cell_shape)))
        global_border_mat = np.zeros((num_of_eqs, num_of_vars * num_of_cells))
        global_border_right = np.zeros(num_of_eqs)
        # print('glob bord', global_border_mat.shape)
        # global_border_mat = np.array([])
        # global_border_right = []

        num_of_connect = len(connect_points)* len(connect_ops)
        num_of_eqs = len(all_cells)*num_of_connect
        
        # global_connect_mat = np.zeros((num_of_eqs, num_of_vars * num_of_cells))
        global_connect_mat = []

        global_connect_right = np.zeros(num_of_eqs)
        
        colloc_left_operators, colloc_right_operators = colloc_ops
        border_left_operators, border_right_operators = border_ops


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

            # global_border_mat.append(border_mat)
            # global_border_right.append(border_r)

            global_colloc_mat[cell_ind * num_of_collocs:(cell_ind+1) * num_of_collocs, cell_ind * num_of_vars:(cell_ind+1) * num_of_vars] = colloc_mat
            global_colloc_right[cell_ind * num_of_collocs:(cell_ind+1) * num_of_collocs] = colloc_r

            # num_of_border = len(border_points_for_use)
            # global_border_mat = concat(global_border_mat, border_)
            # print(border_mat.shape)
            #
            #  global_border_mat.append(border_mat)
            
            # global_border_mat = concat(global_border_mat, border_mat)
            # concat(global_border_mat, border_mat)
            # global_border_right.append(border_r)

            num_of_border = border_mat.shape[0]
            global_border_mat[cell_ind * num_of_border:(cell_ind+1) * num_of_border, cell_ind * num_of_vars:(cell_ind+1) * num_of_vars] = border_mat
            global_border_right[cell_ind * num_of_border:(cell_ind+1) * num_of_border] = border_r

            left_connect_for_use = np.array([np.logical_and(point == -1, ~left_borders).any() for point in connect_points])
            right_connect_for_use = np.array([np.logical_and(point == 1, ~right_borders).any() for point in connect_points])
            connect_points_for_use = connect_points[np.logical_or(left_connect_for_use, right_connect_for_use)]

            # print('connect_points', connect_points_for_use)
            # connect_points_for_use = connect_points[right_connect_for_use] 
            # print('border', border_points_for_use, '\n connect', connect_points_for_use)
        
            connect_mat = self.generate_connection_couple([connect_left_operators, connect_right_operators], cell_num, connect_points_for_use)
            connect_weight = 1
            # num_of_connect = connect_mat.shape[0]
            # global_connect_mat[cell_ind * num_of_connect:(cell_ind+1) * num_of_connect, cell_ind * num_of_vars:(cell_ind+1) * num_of_vars] = connect_mat
            if first_connect:
                # print('initing')
                global_connect_mat = connect_mat
                first_connect = False
            else:
                # print('concating')
                global_connect_mat = concat(global_connect_mat, connect_mat)
            # print('concat_len', (global_connect_mat.shape))
            # print(global_connect_mat)
            # global_connect_mat.append(connect_mat)
            # print(connect_mat, connect_mat.shape)
        

        global_connect_mat = np.array(global_connect_mat)
        global_connect_right = np.zeros(len(global_connect_mat))

            # global_connect_right = connect_r
        # print('colloc ', colloc_mat,'\nborder ', border_mat,'\nconnect ', connect_mat)
        # res_mat = concat(concat(colloc_mat, border_mat), connect_mat * connect_weight)
        # res_right = concat(concat(colloc_r, border_r), connect_r * connect_weight)

        # connect_weight = 1
        # print('COLLOCS \n',global_colloc_mat)
        # print('BORDER \n', global_border_mat)
        # print('CONNECTS \n')
        # print(global_connect_mat)
        # print(global_colloc_right.shape, global_connect_right.shape, global_border_right.shape)
        # global_integral_mat = np.array([self.generate_integral(0)])
        # global_integral_right = np.array([1])
        # for time in range(1, self.dim_sizes[0]):
        #     integral_line = np.array([self.generate_integral(time)])
        #     global_integral_mat = concat(global_integral_mat, integral_line)

        #     global_integral_right = concat(global_integral_right, np.array([1]))
        
        # res_mat = concat(concat(concat(global_colloc_mat, global_border_mat), global_connect_mat * connect_weight), global_integral_mat)
        # res_right = concat(concat(concat(global_colloc_right, global_border_right), global_connect_right * connect_weight), global_integral_right)

        res_mat = concat(concat(global_colloc_mat, global_border_mat), global_connect_mat * connect_weight)
        res_right = concat(concat(global_colloc_right, global_border_right), global_connect_right * connect_weight)

        notnull_ind = np.sum(res_mat != 0, axis=1)!=0
        res_mat = res_mat[notnull_ind]
        res_right = res_right[notnull_ind]
        # print(res_mat,'\n-------\n', res_right, '\n\n')
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