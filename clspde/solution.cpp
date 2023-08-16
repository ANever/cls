#include <iostream>
#include <vector>
#include <cmath>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "basis.hpp"


// #include "qr_solver.hpp"

// def lp(line, function_list, variable_list):
//     splited = line.split(' ')

//     ops_stack = ()

//     def is_der_operator(string: str):
//         if re.findall('\(d\/d..?\)', string):
//             return true
//         else:
//             return false
        
//     def apply_ops(ops_stack: list, func: str):
//         dif_powers = (0)*len(variable_list)
//         for op in ops_stack:
//             op = op.replace('(d/d', '')
//             op = op.replace(')', '')
//             op = op.split('^')

//             var_index = variable_list.index(op(0))
//             try:
//                 power = op(1)
//             except:
//                 power = 1
//             dif_powers(var_index) = int(power)
//         if func(0)=='&':
//             f_name = 'u_loc'
//         else:
//             f_name = 'u_bas'
//         func_index = function_list.index(func.replace('&',''))
//         return (f_name+'('+str(dif_powers)+', '+str(func_index)+')')

//     def is_func(string:str):
//         if string(0)=='&' and (string(1:) in function_list):
//             return (true, 'local')
//         if string in function_list:
//             return (true, 'basis')
//         else:
//             return (false, None)

//     res = ''
//     for i in range(len(splited)):
//         if is_der_operator(splited(i)):
//             ops_stack.append(splited(i))
//         elif is_func(splited(i))(0):
//             res += (apply_ops(ops_stack, splited(i),))
//             ops_stack = ()
//         else:
//             res += splited(i)
//     return res

// def concat(a:np.ndarray, b:np.ndarray):
//     a = np.array(a)
//     b = np.array(b)
//     if b.size == 0:
//         return a
//     if a.size == 0:
//         return a
//     else:
        // return np.concatenate((a, b))

// def f_collocation_points(N):
//     points = np.zeros(N+1)
//     h = 2/(N+1)
//     points(0) = -1 + h/2
//     for i in range(1, N+1):
//         points(i) = points(i-1) + h
//     return np.array(points).reshape(N+1,1)

class Solution{
    /* Solution of pde system

    Main class, that creates linear system of equations from pde and solves it

    Attributes:
        n_dims: amount of arguments of solution funciton
        dim_sizes: amount of cells in every direction (cell grid is square)
        area_lims: list of limitis of area in every dimention
        power: number of basis elements for every dimention
        basis: basis class to create basis elements
        n_funcs: amount of solution funtions

        cells_coefs: coefs of solution decomposition, must be addresed as (func_num, cell_addres, x1_power, x2_power..., xn_power)
    */

public:
    unsigned long n_dims;
    unsigned long power;
    unsigned long n_funcs;
    xt::xarray<double> dim_sizes;
    xt::xarray<double> area_lims;
    xt::xarray<double> steps;
    xt::xarray<double> cells_shape;
    xt::xarray<double> cells_coefs;
    xt::xarray<double> cell_size;
    xt::xarray<double> points;

    xt::xarray<double> basis_evaled;
    xt::xarray<double> basis_evaled_raveled;
    Basis basis;
    
    bool split_mats_inited;
    bool precalculated_basis;

    Solution (unsigned long n_dims, xt::xarray<double> dim_sizes, xt::xarray<double> area_lims,unsigned long power,unsigned long n_funcs){
        this->area_lims = area_lims;
        this->n_dims = n_dims; // = len(dim_sizes);
        this->dim_sizes = dim_sizes; // // n of steps for all directions 
        this->power = power;
        this->n_funcs = n_funcs;
        init_grid();
        this->steps = ((xt::view(this->area_lims, xt::all(), 1) - xt::view(this->area_lims, xt::all(), 0)) / this->dim_sizes);
        
        this->basis = Basis(power, steps=this->steps, n_dims = n_dims);
        
        this->split_mats_inited = false;
        this->precalculated_basis = false;
    }

    void init_grid(void){
        // this->cells_shape = {this->n_funcs, this->dim_sizes, (this->power)*this->n_dims}; //TODO rewrite correctly
        xt::xarray<double> tmp(cells_shape);
        this->cells_coefs = tmp;
        this->cell_size = this->n_funcs * std::pow(this->power, this->n_dims);
    }

    void precalculate_basis(xt::xarray<double> points, unsigned long max_der){
        // this->points = xt::unique(points); // it is in docs
        unsigned long num_points = points.shape()(0);
        xt::xarray<unsigned long>::shape_type tmp_shape = {num_points, max_der + 1, max_der + 1, this->n_dims, this->power};
        xt::xarray<double> tmp(tmp_shape);
        this->basis_evaled = tmp;
        xt::xarray<double>::shape_type tmp_shape = {num_points, max_der + 1, max_der + 1, std::pow(double(this->n_dims), int(this->power))};
        xt::xarray<double> tmp(tmp_shape);
        this->basis_evaled_raveled = tmp;

        for (int i=0; i<points.size();i++){
            // xt::xarray<double> point;
            auto point = points(i);
            for(unsigned long dt=0; dt <= max_der; dt++){
                for(unsigned long dx=0; dx < max_der; dx++){
                    xt::xarray<unsigned long> derivatives = {dt,dx};
                    this->basis_evaled(point_num, dt, dx) = this->basis.eval(point, derivatives, false);
                    xt::xarray<double> be0 = this->basis_evaled(point_num, dt,dx, 0);
                    xt::xarray<double> be1 = this->basis_evaled(point_num, dt,dx, 1);
                    this->basis_evaled_raveled(point_num, dt,dx) = xt::ravel(xt::linalg::outer(be0, be1));
                } 
            }
        }
        this->precalculated_basis = true;
    }

    xt::xarray<unsigned long> point_num(xt::xarray<float> point, float threshold = 1e-3):
        //returns point addres in precalculated basis
        return xt::where(this->points==point)(0,0)
        
    xt::xarray<unsigned long> def localize(global_point: np.ndarray, cells_closed_right: bool = false) -> np.ndarray:
        //returns cell addres and local coordinates in this cell of a globall coordinates
        if cells_closed_right:
            xt::xarray<unsigned long> shift = ((global_point - xt::view(this->area_lims, xt::all(),0)) % this->steps < 1e-10)
            xt::xarray<unsigned long> cell_num = (np.floor((global_point - xt::view(this->area_lims, xt::all(),0)) / this->steps) - shift, dtype=int)
        else:
            cell_num = np.array(np.floor((global_point - xt::view(this->area_lims, xt::all(),0)) / this->steps), dtype=int)

        local_point = 2 * ((np.array(global_point)-xt::view(this->area_lims, xt::all(),0))/np.array(this->steps) - (np.array(cell_num) + 0.5))

        return np.array((cell_num, local_point))

    xt::xarray<double> def globalize(xt::xarray<unsigned long> cell_num, xt::xarray<double> local_point){
        //returns globall coordinates of a point from its cell adress and local coordinates
        global_point = xt::view(this->area_lims, xt::all(),0) + (((local_point) + 1) + 2*(cell_num)) * this->steps/2
        return global_point
    }

    void init_split_mats(void){
        //creates matrices for cell division
        this->minus_shift = np.zeros((this->power, this->power))
        this->plus_shift = np.zeros((this->power, this->power))
        for i in range(this->power):
            for j in range(i):
                this->minus_shift(i,j) = comb(i,j) * (-2)**i
        this->plus_shift = np.abs(this->minus_shift)

        this->split_mats_inited = true
    }
    
    void cell_division(unsigned long dimention = 0){
        //divides cells in half in certain dimention with saving of solution
        
        plus_shift_mat = np.zeros((this->power,this->power))
        minus_shift_mat = np.zeros((this->power,this->power))
        for i in range(this->power):
            for j in range(i+1):
                plus_shift_mat(i,j) = comb(i,j)*2**(-(i))
                minus_shift_mat(i,j) = comb(i,j)*2**(-(i))*(-1)**(i+j)
        inds = (list(range(size)) for size in this->dim_sizes)
        old_cells_inds = list(itertools.product(*inds))
        old_cells_coefs = copy.deepcopy(this->cells_coefs)

        this->dim_sizes(dimention) *= 2
        this->steps = ((this->area_lims(:,1) - this->area_lims(:,0)) / this->dim_sizes)
        this->init_grid()
        cell_index_adder = np.zeros(this->n_dims, int)
        cell_index_adder(dimention)//= 1
        
        for func in range(this->n_funcs):
            for cell in old_cells_inds:
                if dimention%2:
                    fst_cell = tuple(2**cell_index_adder(i) * cell(i) for i in range(this->n_dims))
                    this->cells_coefs(func,fst_cell) = np.tensordot(old_cells_coefs(func,cell), minus_shift_mat, axes=((dimention),(0)))
                    sec_cell = tuple(2**cell_index_adder(i) * cell(i) + cell_index_adder(i) for i in range(this->n_dims))
                    this->cells_coefs(func,sec_cell) = np.tensordot(old_cells_coefs(func,cell), plus_shift_mat, axes=((dimention),(0)))
                else: //transpose if needed
                    fst_cell = tuple(2**cell_index_adder(i) * cell(i) for i in range(this->n_dims))
                    this->cells_coefs(func,fst_cell) = np.tensordot(minus_shift_mat,old_cells_coefs(func,cell), axes=((0),(dimention)))
                    sec_cell = tuple(2**cell_index_adder(i) * cell(i) + cell_index_adder(i) for i in range(this->n_dims))
                    this->cells_coefs(func,sec_cell) = np.tensordot(plus_shift_mat, old_cells_coefs(func,cell), axes=((0),(dimention))) 
    }

    // void powerup(void){
    //     //raises power(amount of basis elements) of solution without change in it
    //     rows=np.array(range(this->power))
    //     columns = rows

    //     this->power = this->power + 1
    //     old_cell_coefs = copy.deepcopy(this->cells_coefs)
    //     this->init_grid()

    //     this->Basis=Basis(this->power + 1, steps=this->steps, n_dims = this->n_dims)

    //     inds = (list(range(size)) for size in this->dim_sizes)
    //     old_cells_inds = list(itertools.product(*inds))

    //     for func in range(this->n_funcs):
    //         for cell in old_cells_inds:
    //             this->cells_coefs(func,cell,rows(:,np.newaxis),columns) = old_cell_coefs(func,cell)
    // }

    double eval(xt::xarray<double> point,
                xt::xarray<unsigned long> derivatives,
                xt::xarray<unsigned long> cell_num,
                unsigned long func = 0, 
                bool local = false, 
                bool cells_closed_right = false){
        /*evaluation of solution function with argument x and list of partial derivatives
        
        Args:
            x: point to evaluate solution in, np.array(n_dim, float)
            derivatives: list of derivatives to take before evalution, np.array(n_dim, int)
            func: number of a solution function to evaluate
            local: whether x is local or not
            cell_num: cell addres of a local x
        */
        derivatives = xt::abs(derivatives);
        
        xt::xarray<double> local_point;
        if(local){
            local_point = point;
        }
        else{
            cell_num, local_point = localize(point, cells_closed_right);
        }
        
        coefs = this->cells_coefs(tuple(np.insert(np.array(cell_num, dtype=int), 0, func)))
        result = copy.deepcopy(coefs)
        //applying coefs tensor to evaled basis in point
        if this->precalculated_basis:
            try:
                point_num = this->point_num(local_point)
                basis_evaled = this->basis_evaled(point_num,tuple(derivatives))
            except IndexError: //actually i forgot what error should be here
                pass
        else:
            basis_evaled = this->basis.eval(local_point, derivatives, ravel=false)
        for b_e in basis_evaled(::-1):
            result = result.dot(b_e)
        return result
    }

    void prepare_ops(list_of_ops, function_list, variable_list):
        for i in range(len(list_of_ops)):
            op = list_of_ops(i)
            if isinstance(op, str):
                list_of_ops(i) = lp(op, function_list, variable_list)

    def generate_system(cell_num: np.ndarray, points: np.ndarray, colloc_ops, border_ops, connect_ops = (),
                        function_list = ('u', 'v'), variable_list=('x','y')) -> tuple:
        colloc_points, connect_points, border_points = points

        def dir(point: np.ndarray) -> np.ndarray:
            direction = (np.abs(point) == 1) * (np.sign(point))
            return np.array(direction, dtype=int)

        w = (this->steps(0)/2) // weight
        // default connection
        if len(connect_ops) == 0:
            connect_left_operators = ()
            connect_right_operators = ()
            for func_num in range(this->n_funcs):
                // func_num=func_num is carrying that is needed to distinguish labmdas
                connect_left_operators += (lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) + np.sum(dir(x_loc)) * u_bas(dir(x_loc),func_num) * w,
                                           lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(2*dir(x_loc),func_num)* w**2 + np.sum(dir(x_loc)) * u_bas(3*dir(x_loc),func_num)* w**3)
                connect_right_operators += (lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) + np.sum(dir(x_loc))*u_bas(dir(x_loc),func_num)* w,
                                            lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(2*dir(x_loc),func_num) * w**2 + np.sum(dir(x_loc)) * u_bas(3*dir(x_loc),func_num)* w**3)

            connect_ops = (connect_left_operators, connect_right_operators)

        //default colloc points
        if len(colloc_points) == 0:
            colloc_points = f_collocation_points(this->power)
        
        connect_left_operators, connect_right_operators = connect_ops

        colloc_left_operators, colloc_right_operators = colloc_ops

        border_left_operators, border_right_operators = border_ops

        for ops in (connect_left_operators, connect_right_operators, colloc_left_operators, colloc_right_operators, border_left_operators, border_right_operators):
            this->prepare_ops(ops, function_list, variable_list)

        colloc_mat, colloc_r = this->generate_subsystem(colloc_left_operators, colloc_right_operators, cell_num, colloc_points)
        
        left_borders = cell_num == np.zeros(this->n_dims)
        right_borders = cell_num == (this->dim_sizes-1)
        
        left_border_for_use = np.array((np.logical_and(point == -1, left_borders).any() for point in border_points))
        right_border_for_use = np.array((np.logical_and(point == 1, right_borders).any() for point in border_points))
        border_points_for_use = border_points(np.logical_or(left_border_for_use, right_border_for_use))

        border_mat, border_r = this->generate_subsystem(border_left_operators, border_right_operators, cell_num, border_points_for_use)
        
        left_connect_for_use = np.array((np.logical_and(point == -1, ~left_borders).any() for point in connect_points))
        right_connect_for_use = np.array((np.logical_and(point == 1, ~right_borders).any() for point in connect_points))
        connect_points_for_use = connect_points(np.logical_or(left_connect_for_use, right_connect_for_use)) 
        
        connect_mat, connect_r = this->generate_subsystem(connect_left_operators, connect_right_operators, cell_num, connect_points_for_use)
        connect_weight = 1

        // print('normalized')
        // def normalize(mat, r):
        //     coef = np.mean(mat(mat!=0))
        //     mat /= coef
        //     r /= coef
        //     return mat, r
        
        // for (mat, r) in zip((colloc_mat, connect_mat, border_mat),(colloc_r, connect_r, border_r)):
        //     mat ,r = normalize(mat, r)

        res_mat = concat(concat(colloc_mat, border_mat), connect_mat * connect_weight)
        res_right = concat(concat(colloc_r, border_r), connect_r * connect_weight)

        return res_mat, res_right

    def iterate_cells(solver='np', **kwargs) -> None:
        //solves local system of equations for every cell once
        inds = (list(range(size)) for size in this->dim_sizes)
        all_cells = list(itertools.product(*inds))
        cell_shape = tuple((this->power)*this->n_dims)
        for cell in all_cells:
            mat, right = this->generate_system(cell, **kwargs)
            cell_size = np.prod(cell_shape)
            solution = this->_solver(A = mat, b = right, solver=solver)
            for i in range(this->n_funcs):
                this->cells_coefs((i,*cell)) = solution(i*cell_size:(i+1)*cell_size).reshape(cell_shape)

    def _solver(A, b, solver = 'np', svd_threshold = 1e-5, verbose = false):
        //solves system Ax=b with chosen solver//
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

    def solve(threshold = 1e-5, max_iter = 10000, verbose=false, **kwargs) -> None:
        /*solves set problem with iterative solution of local systems in every cell 
        untill convergence
        */
        prev_coefs = copy.deepcopy(this->cells_coefs)
        for i in range(max_iter):
            this->iterate_cells(**kwargs)
            residual = np.max(np.abs((prev_coefs - this->cells_coefs)))
            if verbose:
                print(residual)
            if residual < threshold:
                break
            prev_coefs = copy.deepcopy(this->cells_coefs)
        if verbose:
            print('Iterations to converge: ', i)

    def generate_integral(time):
        //TODO not implemented yet, for integral conditions
        //generate common line
        n = this->power
        integral_cell = 1/np.array(range(1,n+1)) * ((2, 0)*int(np.ceil(n/2)))(:n)
        full_line = np.zeros(np.prod(this->cells_shape))
        a = np.zeros((this->power, this->power))
        a(0) = integral_cell
        cell_line = concat(np.ravel(a), np.ravel(a) * 0)

        inds = (list(range(size)) for size in this->dim_sizes)
        all_cells = list(itertools.product(*inds))
        num_of_vars = this->cell_size //will work only for 2d
        for cell_num in all_cells:
            if cell_num(0) == time:
                cell_ind = cell_num(1) + cell_num(0) * this->dim_sizes(1)
                full_line(cell_ind * num_of_vars:(cell_ind+1) * num_of_vars) = cell_line
        return full_line
        //TODO
        //set time moment
        //iterate over space
        //set common line into cells
        

    def generate_eq(cell_num, left_side_operator, right_side_operator, points):
        //basic func for generating equation from leftside and rightside operators
        def left_side(operator, cell_num, point: np.ndarray) -> np.ndarray:
            //must return row of coeficient for LSE//
            loc_point = copy.deepcopy(point)
            global_point = this->globalize(cell_num, point)
            x = copy.deepcopy(global_point)
            def u_bas(der,func=0):
                bas_size = int(this->cell_size/this->n_funcs)
                result = np.zeros(this->n_funcs * bas_size)
                if this->precalculated_basis:
                    loc_point_num = this->point_num(loc_point)
                    result(func*bas_size:(func+1)*bas_size) = this->basis_evaled_raveled(loc_point_num, der(0),der(1))
                else:
                    result(func*bas_size:(func+1)*bas_size) = this->basis.eval(loc_point, der, ravel=true)
                return result

            def u_loc(der,func=0):
                try:
                    result = this->eval(loc_point, der, func=func, local = true, cell_num = cell_num)
                except IndexError:
                    result = this->eval(loc_point, der, func=func, local = true, cell_num = cell_num, cells_closed_right=true)
                return result

            return operator(u_loc, u_bas, x, loc_point)

        def right_side(operator, cell_num, point: np.ndarray) -> float:
            
            def dir(point: np.ndarray) -> np.ndarray:
                direction = (np.abs(point) == 1) * (np.sign(point)) 
                return direction

            global_point = this->globalize(cell_num, point)
            // x = global_point
            loc_point = copy.deepcopy(point)            
            u_loc = lambda der, func_num=0: this->eval(loc_point, der, local = true, cell_num = cell_num, func=func_num)   // for linearization purpses

            neigh_point = loc_point-2*dir(loc_point)

            u_nei = lambda der, func_num=0: this->eval(neigh_point, der, local = true, cell_num = cell_num + dir(loc_point), func=func_num)
            return operator(u_loc, u_nei, global_point, loc_point) //x
        
        mat = np.zeros((len(points), this->cell_size))
        r_side = np.zeros((len(points)))
        for i in range(len(points)):
            mat(i) = left_side(left_side_operator, cell_num,  points(i))
            r_side(i) = right_side(right_side_operator, cell_num, points(i))
        return mat, r_side

    def generate_subsystem(left_ops, right_ops, cell_num, points: np.ndarray) -> tuple:
        mat, r = this->generate_eq(cell_num, left_ops(0), right_ops(0), points)
        for i in range(1,len(left_ops)):
            mat_small, r_small = this->generate_eq(cell_num, left_ops(i), right_ops(i), points)
            mat = concat(mat, mat_small)
            r = concat(r, r_small)
        return mat, r

    def generate_connection_couple(left_ops, cell_num, points: np.ndarray) -> tuple:
        // left ops must be a pair of functions
        // right ops substitude
        right_ops = (lambda u, _, x, x_loc: 0) * len(left_ops(0))
        def dir(point: np.ndarray) -> np.ndarray:
            direction = (np.abs(point) == 1) * (np.sign(point))
            return np.array(direction, dtype=int)

        first = true
        for point in points:
            first_line, _ = this->generate_subsystem(left_ops(0), right_ops, cell_num, np.array((point)))
            neigh = tuple(np.array(cell_num) + dir(point))
            neigh_point = point - 2*dir(point)
            second_line, _ = this->generate_subsystem(left_ops(1), right_ops, neigh, np.array((neigh_point)))

            connect_line = np.zeros((len(left_ops(0)), np.prod(this->cells_coefs.shape)))

            index = this->cell_index(cell_num)
            neigh_index = this->cell_index(neigh)

            connect_line(:, index*this->cell_size:(index+1)*this->cell_size) = first_line
            connect_line(:, neigh_index*this->cell_size:(neigh_index+1)*this->cell_size) = -second_line
            if first:
                connect_mat = connect_line
                first = false
            else:
                connect_mat = concat(connect_mat, connect_line)
            // connect_mat.append(connect_line) //???
        return np.array(connect_mat)

    def plot(n = 100):
        func = np.zeros(n)
        grid = np.linspace(this->area_lims(0,0), this->area_lims(0,1), n, endpoint=false)
        for i in range(len(grid)): 
            func(i) = this->eval(grid(i), (0))
        plt.plot(func)
        plt.show()
    
    def plot2d(n=100, x_lims = None, y_lims = None, func_num=0, derivatives = (0,0)):
        func = np.zeros((n,n))
        if x_lims == None:
            x_lims = this->area_lims(0)
        if y_lims == None:
            y_lims = this->area_lims(1)
        ax1 = np.linspace(x_lims(0), x_lims(1), n, endpoint=false)
        ax2 = np.linspace(y_lims(0), y_lims(1), n, endpoint=false)
        X, Y = np.meshgrid(ax1, ax2)

        for i in range(n):
            for j in range(n): 
                func(j, i) = this->eval((ax1(i), ax2(j)), derivatives, func=func_num)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize = (7,7))
        surf = ax.plot_surface(X, Y, func, cmap=cm.coolwarm,
                        linewidth=0, antialiased=false)

        // ax.set_xticks(X)
        // ax.set_xticks(Y)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        plt.show()


    def cell_index(cell_num):
        if this->n_dims == 2:
            cell_ind = cell_num(0) + cell_num(1) * this->dim_sizes(0) // + cell_num(2) * prod(this->dim_sizes(:2)...
        elif this->n_dims == 1:
            cell_ind = cell_num(0)
        else:
            raise LookupError
        return cell_ind

    def generate_global_system(points: np.ndarray, colloc_ops, border_ops, connect_ops = (), connect_weight=1, 
                               function_list = ('u', 'v'), variable_list=('x','y')) -> tuple:
        
        colloc_points, connect_points, border_points = points

        def dir(point: np.ndarray) -> np.ndarray:
            direction = (np.abs(point) == 1) * (np.sign(point))
            return np.array(direction, dtype=int)

        w = (this->steps(0)/2)//weight
        //default connection
        if len(connect_ops) == 0:
            connect_left_operators = ()
            connect_right_operators = ()
            for func_num in range(this->n_funcs):
                //func_num=func_num is carrying that is needed to distinguish lambdas
                
                connect_left_operators += (lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) + np.sum(dir(x_loc)) * u_bas(dir(x_loc),func_num) * w,
                                            lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(2*dir(x_loc),func_num)* w**2 + np.sum(dir(x_loc)) * u_bas(3*dir(x_loc),func_num)* w**3)
                connect_right_operators += (lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) - np.sum(dir(x_loc))*u_bas(dir(x_loc),func_num)* w,
                                            lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(2*dir(x_loc),func_num) * w**2 - np.sum(dir(x_loc)) * u_bas(3*dir(x_loc),func_num)* w**3)
            connect_ops = (connect_left_operators, connect_right_operators)

        //default colloc points
        if len(colloc_points) == 0:
            colloc_points = f_collocation_points(this->power)
        
        connect_left_operators, connect_right_operators = connect_ops

        num_of_vars = this->cell_size // np.prod(this->cells_coefs.shape)

        inds = (list(range(size)) for size in this->dim_sizes)
        all_cells = list(itertools.product(*inds))
        
        num_of_collocs = len(colloc_points) * len(colloc_ops(0))
        num_of_eqs = len(all_cells) * num_of_collocs
        num_of_cells = len(all_cells)

        global_colloc_mat = np.zeros((num_of_eqs, num_of_vars * num_of_cells))
        global_colloc_right = np.zeros(num_of_eqs)

        num_of_border = len(border_points)* len(border_ops(0))
        num_of_eqs = len(all_cells)*num_of_border

        global_border_mat = np.zeros((num_of_eqs, num_of_vars * num_of_cells))
        global_border_right = np.zeros(num_of_eqs)
        
        num_of_connect = len(connect_points)* len(connect_ops)
        num_of_eqs = len(all_cells)*num_of_connect
        
        global_connect_mat = ()
        global_connect_right = np.zeros(num_of_eqs)
        
        colloc_left_operators, colloc_right_operators = colloc_ops
        border_left_operators, border_right_operators = border_ops

        for ops in (connect_left_operators, connect_right_operators, colloc_left_operators, colloc_right_operators, border_left_operators, border_right_operators):
            this->prepare_ops(ops, function_list, variable_list)

        first_connect = true
        for cell_num in all_cells:
            left_borders = cell_num == np.zeros(this->n_dims)
            right_borders = cell_num == (this->dim_sizes-1)
            left_border_for_use = np.array((np.logical_and(point == -1, left_borders).any() for point in border_points))
            
            right_border_for_use = np.array((np.logical_and(point == 1, right_borders).any() for point in border_points))
            border_points_for_use = border_points(np.logical_or(left_border_for_use, right_border_for_use))

            colloc_mat, colloc_r = this->generate_subsystem(colloc_left_operators, colloc_right_operators, cell_num, colloc_points)
            border_mat, border_r = this->generate_subsystem(border_left_operators, border_right_operators, cell_num, border_points_for_use)

            //for 2d only!
            // cell_ind = cell_num(0) + cell_num(1) * this->dim_sizes(0) // + cell_num(2) * prod(this->dim_sizes(:2)...
            cell_ind = this->cell_index(cell_num)
            global_colloc_mat(cell_ind * num_of_collocs:(cell_ind+1) * num_of_collocs, cell_ind * num_of_vars:(cell_ind+1) * num_of_vars) = colloc_mat
            global_colloc_right(cell_ind * num_of_collocs:(cell_ind+1) * num_of_collocs) = colloc_r
            
            num_of_bordto_use = border_mat.shape(0)
            global_border_mat(cell_ind * num_of_border:cell_ind * num_of_border + num_of_bordto_use, cell_ind * num_of_vars:(cell_ind+1) * num_of_vars) = border_mat
            global_border_right(cell_ind * num_of_border:cell_ind * num_of_border + num_of_bordto_use) = border_r

            left_connect_for_use = np.array((np.logical_and(point == -1, ~left_borders).any() for point in connect_points))
            right_connect_for_use = np.array((np.logical_and(point == 1, ~right_borders).any() for point in connect_points))
            connect_points_for_use = connect_points(np.logical_or(left_connect_for_use, right_connect_for_use))
            
            connect_mat = this->generate_connection_couple((connect_left_operators, connect_right_operators), cell_num, connect_points_for_use)
            if first_connect:
                global_connect_mat = connect_mat
                first_connect = false
            else:
                global_connect_mat = concat(global_connect_mat, connect_mat)
        global_connect_mat = np.array(global_connect_mat)
        global_connect_right = np.zeros(len(global_connect_mat))

        // def normalize(mat, r):
        //     coef = np.max(np.abs(mat(mat!=0)))
        //     mat /= coef
        //     r /= coef
        //     return mat, r
        
        // for (mat, r) in zip((global_colloc_mat, global_connect_mat, global_border_mat),(global_colloc_right, global_connect_right, global_border_right)):
        //     mat ,r = normalize(mat, r)

        res_mat = concat(concat(global_colloc_mat, global_border_mat), global_connect_mat)
        res_right = concat(concat(global_colloc_right, global_border_right), global_connect_right)

        notnull_ind = np.sum(res_mat != 0, axis=1)!=0
        res_mat = res_mat(notnull_ind)
        res_right = res_right(notnull_ind)

        return res_mat, res_right

    def global_solve(solver = 'np',return_system = false, calculate = true, svd_threshold = 1e-4, verbose = false, alpha=0,  **kwargs):
        A, b = this->generate_global_system(**kwargs)

        if alpha > 0:
            A = concat(A, np.eye(A.shape(1))*alpha)
            b = concat(b, np.zeros(A.shape(1)))
        if calculate:
            res = this->_solver(A,b,solver=solver, svd_threshold=svd_threshold, verbose = verbose)
            inds = (list(range(size)) for size in this->dim_sizes)
            all_cells = list(itertools.product(*inds))

            cell_shape = tuple((this->power)*this->n_dims)
            cell_size = np.prod(cell_shape)
            size = int(cell_size*this->n_funcs)

            for cell in all_cells:
                cell_index = this->cell_index(cell)
                cell_res = res(size*cell_index:size*(cell_index+1))

                for i in range(this->n_funcs):
                    this->cells_coefs((i,*cell)) = cell_res(i*cell_size:(i+1)*cell_size).reshape(cell_shape)    
        if return_system:
            return A, b
    
}

// ______________________________TESTING________________________

// if __name__ == '__main__':
    
//     def f_collocation_points(N):
//         points = np.zeros(N+1)
//         h = 2/(N+1)
//         points(0) = -1 + h/2
//         for i in range(1, N+1):
//             points(i) = points(i-1) + h
//         return np.array(points).reshape(N+1,1)
//     colloc_points = f_collocation_points(4)


//     power = 5
//     params = {
//         'n_dims': 1,
//         'dim_sizes': np.array((5)),
//         'area_lims': np.array(((0,1))),
//         'power': power,
//         'basis': Basis(power),
//     }
//     sol = Solution(**params)

//     w = (sol.steps(0)/2)

//     colloc_left_operators = (lambda u_loc, u_bas, x, x_loc: u_bas((4)) * (w**4))
//     colloc_right_operators = (lambda u_loc, u_nei, x, x_loc: np.exp(x)*(x**4 + 14*(x**3) + 49*(x**2) + 32*x - 12) * (w**4))
//     colloc_ops = (colloc_left_operators, colloc_right_operators)

//     border_left_operators = (lambda _, u_bas, x, x_loc: u_bas((0)), 
//                                 lambda _, u_bas, x, x_loc: u_bas((1)) * w)
//     border_right_operators = (lambda u, _, x, x_loc: 0,
//                                 lambda u, _, x, x_loc: 0 * w)
//     border_ops = (border_left_operators, border_right_operators)

//     connect_points = np.array(((-1), (1)))
//     border_points = connect_points

//     points = (colloc_points, connect_points, border_points)

//     iteration_dict = {'points':points,
//                     'colloc_ops':colloc_ops,
//                     'border_ops':border_ops}

//     sol.iterate_cells(**iteration_dict)

//     print('executed')