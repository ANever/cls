from threading import local
import numpy as np
import copy
import itertools
from .basis import Basis
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt

from qr_solver import QR_solve

def concat(a:np.array, b:np.array):
    if b.size == 0:
        return a
    if a.size == 0:
        return a
    else:
        return np.concatenate((a, b))
        
class Solution():

    def __init__(self, n_dims: int, dim_sizes: np.array, area_lims: np.array, power:int, basis: Basis) -> None:
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
        self.init_grid()
        self.steps = ((self.area_lims[:,1] - self.area_lims[:,0]) / self.dim_sizes)
        
        self.basis = Basis(power, steps=self.steps)

        

    def init_grid(self) -> None:
        cells_shape = tuple(list(self.dim_sizes) + [self.power]*self.n_dims)
        self.cells_coefs = np.zeros(cells_shape)
        self.cell_size = self.power**self.n_dims
        
    def localize(self, global_point: np.array, cells_closed_right: bool = False) -> np.array:
        if cells_closed_right:
            shift = np.array((global_point % self.steps) < 1e-12, dtype=int)
            cell_num = np.array(np.floor(global_point / self.steps) - shift, dtype=int)
        else:
            cell_num = np.array(np.floor(global_point / self.steps), dtype=int)

        #local_point = copy.deepcopy()
        local_point = np.array(global_point)/np.array(self.steps) - (cell_num - 0.5) #TODO CHECK
        return cell_num, local_point

    def globalize(self, cell_num: np.array, local_point: np.array) -> np.array:
        global_point = (np.array(local_point) + np.array(cell_num) - 0.5) * self.steps #TODO CHECK
        return global_point


    def eval(self, global_point: np.array, derivatives: np.array, cells_closed_right: bool = False) -> np.float:
        '''
        x - np.array(n_dim, float)
        derivatives - np.array(n_dim, int)
        evaluation of solution function with argument x and list of partial derivatives
        '''
        # if (derivatives < 0).any() :
        #     raise Exception("Negative derivative")
        derivatives = np.abs(derivatives)
        #eval cell, that x is in
        cell_num, local_point = self.localize(global_point, cells_closed_right)
        coefs = self.cells_coefs[cell_num]
        result = copy.deepcopy(coefs)
        #applying coefs tensor to evaled basis in point
        basis_evaled = self.basis.eval(local_point, derivatives)
        for i in range(self.n_dims):
            result = coefs @ basis_evaled[i]
        return result[0]

    def eval_loc(self, cell_num, local_point, derivatives):
        coefs = self.cells_coefs[cell_num]
        result = copy.deepcopy(coefs)
        basis_evaled = self.basis.eval(local_point, derivatives)
        for i in range(self.n_dims):
            result = result @ basis_evaled[i]
        return result[0]

    def generate_system(self, cell_num: np.array, points: np.array) -> tuple:
        colloc_points, connect_points, border_points = points

        #TEMP TODO move operators grom here to be set by user
        #TEMP TODO move operators grom here to be set by user
        #TEMP TODO move operators grom here to be set by user


        colloc_left_operators = [lambda _, u, x: u([4]) * ((self.steps[0]/2)**4)]
        colloc_right_operators = [lambda u, _, x: np.exp(x)*(x**4 + 14*(x**3) + 49*(x**2) + 32*x - 12) * ((self.steps[0]/2)**4)]

        border_left_operators = [lambda _, u, x: u([0]), 
                                    lambda _, u, x: u([1]) * (self.steps[0]/2)]
        border_right_operators = [lambda u, _, x: 0, 
                                    lambda u, _, x: 0]

        def dir(point: np.array) -> np.array:
            direction = (np.abs(point) == 1) * point 
            return direction
        
        connect_left_operators = [lambda _, u, x: u([0]) + np.sum(dir(x))*u(dir(x)) * (self.steps[0]/2),
                                 lambda _, u, x: u(2*dir(x)) * (self.steps[0]/2)**2 + np.sum(dir(x)) * u(3*dir(x))*(self.steps[0]/2) **3]
        connect_right_operators = [lambda _, u, x: u([0]) - np.sum(dir(x))*u(dir(x))* (self.steps[0]/2),
                                 lambda _, u, x: u(2*dir(x))* (self.steps[0]/2) **2 + np.sum(dir(x))*u(3*dir(x))*(self.steps[0]/2) **3]


        #TEMP TODO move operators grom here to be set by user
        #TEMP TODO move operators grom here to be set by user
        #TEMP TODO move operators grom here to be set by user


        colloc_mat, colloc_r = self.generate_subsystem(colloc_left_operators, colloc_right_operators, cell_num, colloc_points)
        #colloc_mat, colloc_r = self.generate_colloc(cell_num, colloc_points)
       
        left_borders = cell_num == np.zeros(self.n_dims)
        right_borders = cell_num == (self.dim_sizes-1)
        
        left_border_for_use = np.array([np.logical_and(point == -1, left_borders).any() for point in border_points])
        right_border_for_use = np.array([np.logical_and(point == 1, right_borders).any() for point in border_points])
        border_points_for_use = border_points[np.logical_or(left_border_for_use, right_border_for_use)]

        border_mat, border_r = self.generate_subsystem(border_left_operators, border_right_operators, cell_num, border_points_for_use)
        #border_mat, border_r = self.generate_border(cell_num, border_points_for_use)
        
        left_connect_for_use = np.array([np.logical_and(point == -1, ~left_borders).any() for point in connect_points])
        right_connect_for_use = np.array([np.logical_and(point == 1, ~right_borders).any() for point in connect_points])
        connect_points_for_use = connect_points[np.logical_or(left_connect_for_use, right_connect_for_use)] 

        # print('border', border_points_for_use, '\n connect', connect_points_for_use)
        
        connect_mat, connect_r = self.generate_subsystem(connect_left_operators, connect_right_operators, cell_num, connect_points_for_use)
        #connect_mat, connect_r = self.generate_connect(cell_num, connect_points_for_use)

        #print('colloc ', colloc_mat,'\nborder ', border_mat,'\nconnect ', connect_mat)
        res_mat = concat(concat(colloc_mat, border_mat), connect_mat)
        res_right = concat(concat(colloc_r, border_r), connect_r)

        print(res_mat,'\n', res_right)
        return res_mat, res_right

    def iterate_cells(self, points) -> None:
        inds = [list(range(size)) for size in self.dim_sizes]
        all_cells = list(itertools.product(*inds))
        cell_shape = tuple([self.power]*self.n_dims)
        for cell in all_cells:
            mat, right = self.generate_system(cell, points)
            self.cells_coefs[cell] = QR_solve(mat, right).reshape(cell_shape)

    def generate_eq(self, cell_num, left_side_operator, right_side_operator, points): #TODO refactor u_loc, u_bas, u_nei choice
        '''
        basic func for generating equation
        '''
        def left_side(operator, cell_num, point: np.array) -> np.array:
            '''must return row of coeficient for LSE'''
            loc_point = copy.deepcopy(point)
            global_point = self.globalize(cell_num, point)
            x = global_point
            u_loc = lambda der: self.eval_loc(cell_num, loc_point, der)   # for linearization purpses
            u_bas = lambda der: self.basis.eval(loc_point, der)
            return operator(u_loc, u_bas, x) #u(1)

        def right_side(operator, cell_num, point: np.array) -> np.float:
            
            def dir(point: np.array) -> np.array:
                direction = (np.abs(point) == 1) * point 
                return direction

            global_point = self.globalize(cell_num, point)
            x = global_point
            loc_point = copy.deepcopy(point)            
            u_loc = lambda der: self.eval_loc(cell_num, loc_point, der)

            neigh_point = loc_point-2*dir(loc_point)
            u_nei = lambda der: self.eval_loc(cell_num - dir(loc_point), neigh_point, der) #neighbour cell for connection eqs
            # i dont understand why -dir(x) works and +dir(x) doesnt
            #u = lambda point, der: self.eval_loc(cell_num, point, der)
            return operator(u_loc, u_nei, x) #x
        
        mat = np.zeros((len(points), self.cell_size))
        r_side = np.zeros((len(points)))
        for i in range(len(points)):
            mat[i] = left_side(left_side_operator, cell_num,  points[i])
            r_side[i] = right_side(right_side_operator, cell_num, points[i])
        return mat, r_side


    def generate_subsystem(self, left_ops, right_ops, cell_num, points: np.array) -> tuple:
        mat, r = self.generate_eq(cell_num, left_ops[0], right_ops[0], points)
        print(mat)
        for i in range(1,len(left_ops)):
            mat_small, r_small = self.generate_eq(cell_num, left_ops[i], right_ops[i], points)
            print(mat_small)
            mat = concat(mat, mat_small)
            r = concat(r, r_small)
        return mat, r
    
    def plot(self):
        n = 1000
        func = np.zeros(n)
        grid = np.linspace(self.area_lims[0,0], self.area_lims[0,1], n, endpoint=False)
        for i in range(len(grid)): 
            func[i] = self.eval(grid[i],[0])
        plt.plot(func)
        plt.show()
'''
    def generate_colloc(self, cell_num, points: np.array) -> tuple:
        colloc_left_operator = lambda _, u, x: u(x,[1])
        colloc_right_operator = lambda u, _, x: x
        colloc_mat, colloc_r = self.generate_eq(cell_num, colloc_left_operator, colloc_right_operator, points) 
        return colloc_mat, colloc_r
    
    def generate_border(self,cell_num,  points: np.array) -> tuple:
        border_left_operator = [lambda _, u, x: u(x,[0]), lambda _, u, x: u(x,[1])]
        border_right_operator = [lambda u, _, x: 0, lambda u, _, x: 0]
        border_mat, border_r = self.generate_eq(cell_num, border_left_operator[0], border_right_operator[0], points)
        # for i in range(1,len(border_left_operator)):
        #     border_mat_small, border_r_small = self.generate_eq(cell_num, border_left_operator[i], border_right_operator[i], points)
        #     border_mat = concat(border_mat, border_mat_small)
        #     border_r = concat(border_r, border_r_small)

        return border_mat, border_r

    def generate_connect(self, cell_num, points: np.array) -> tuple:
        def dir(point: np.array) -> np.array:
            direction = (np.abs(point) == 1) * point 
            return direction

        # connect_left_operator = [lambda u, x: u(x) + np.sum(dir(x))*u(x, np.abs(dir(x))),
        #                          lambda u, x: u(x, 2*np.abs(dir(x))) + np.sum(dir(x)) * u(x, 3*np.abs(dir(x)))]
        # connect_right_operator = [lambda u, x: u(x-2*dir(x)) - np.sum(dir(x))*u(x-2*dir(x), np.abs(dir(x))),
        #                          lambda u, x: u(x) + np.sum(dir(x))*u(x,dir)]
        connect_left_operator = [lambda _, u, x: u(x) + np.sum(dir(x))*u(x, dir(x)),
                                 lambda _, u, x: u(x, 2*dir(x)) + np.sum(dir(x)) * u(x, 3*dir(x))
                                 ]
        connect_right_operator = [lambda _, u, x: u(x-2*dir(x), [0]) - np.sum(dir(x))*u(x-2*dir(x), dir(x)),
                                 lambda _, u, x: u(x-2*dir(x), 2*dir(x)) + np.sum(dir(x))*u(x-2*dir(x),3*dir(x))
                                 ]
        connect_mat, connect_r = self.generate_eq(cell_num, connect_left_operator[0], connect_right_operator[0], points)
        # for i in range(1,len(connect_left_operator)):
        #     connect_mat_small, connect_r_small = self.generate_eq(cell_num, connect_left_operator[i], connect_right_operator[i], points)
        #     connect_mat = np.concatenate(connect_mat, connect_mat_small)
        #     connect_r = np.concatenate(connect_r, connect_r_small)

        return connect_mat, connect_r
    # def generate_right_side(self, global_point):'''




if __name__ == '__main__':
    power = 3
    params = {
        'n_dims': 1,
        'dim_sizes': np.array([5]),
        'area_lims': np.array([[0,1]]),
        'power': power,
        'basis': Basis(power),
    }
    sol = Solution(**params)

    colloc_points = np.linspace(-1,1,4,endpoint=True).reshape(4,1)
    connect_points = np.array([[-1], [1]])
    border_points = connect_points

    points = (colloc_points, connect_points, border_points)
    # border_points = [-1, 1]

    sol.iterate_cells(points)