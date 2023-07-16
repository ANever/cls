<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

# CLSPDE

Collocation least squares method library for solving PDE systems.

## Installation:

From source:
```
pip install git+https://github.com/ANever/cls.git
```

**Dependencies:** `numpy`, `matplotlib
`
## How to use

1. Determine problem parameters
```
power = 4
params = {
        'n_dims': 2,
        'dim_sizes': np.array([2, 2]),
        'area_lims': np.array([[0,1], [0,1]]),
        'power': power,
        'basis': Basis(power),
        'n_funcs': 2,
    }
```
2. Initiate solution
```
sol = Solution(**params)
```
3. Initiate equations 
e.g.
```
colloc_left_operators = [lambda u_loc, u_bas, x, x_loc:  (u_bas([1,0],0)-eps*u_bas([0,2],0)
                                                          -(u_bas([0,1],0)*u_loc([0,1],1)
                                                           +u_bas([0,0],0)*u_loc([0,2],1))
                                                          )*colloc_weight,
...
]

colloc_right_operators = [lambda u_loc, u_nei, x, x_loc: (0) * w**2*colloc_weight,
...
]
colloc_ops = [colloc_left_operators, colloc_right_operators]
```
what is equivalent to 
$$ \frac{\patrial}{\partial t} u_1 -\varepsilon \frac{\patrial^2}{\partial x^2} u_1 - div (u_1 \nabla \hat{u_2}) = 0 $$
where $ \hat{u} $ is considered known from previous iteration (initially is 0).

u_loc(derivatives, func_number) - funcitons(x,x_loc), evaluates u from previois iteration, returns scalar
u_bas(derivatives, func_number) - funcitons(x,x_loc), evaluates basis vector, returns vector
u_nei(deribatives, func_number) - funcitons(x,x_loc), evaluates in neighbouring cell (used in connection equations), returns scalar. Acts as u_loc for iterative solve and as u_bas for global solve.
x - global point coordinates to evaluate in
x_loc - local point coordinates to evaluate in

IMPORTANT! left operators must be linear combination of u_bas functions!

4. Determine collocation points
e.g.
```
connect_points = np.array([[-1, 0.5], [1, 0.5],
                            [0.5, -1], [0.5, 1],
                            [-1, -0.5], [1, -0.5],
                            [-0.5, -1], [-0.5, 1],
                            [-1, 0], [1, 0],
                            [0, -1], [0, 1],
                            ])
```
5. Collect all points and equations
```
points=[colloc_points, connect_points ,border_points]
iteration_dict = {'points':points,
                  'colloc_ops':colloc_ops,
                  'border_ops':border_ops,
                  'connect_ops':connect_ops,
}
```
6. Run solve procedures

generate and solve global system of equations
```
sol.global_solve(**iteration_dict)
```
iterate through cells and generate and solve local systems in every cell
```
sol.solve(**iteration_dict)
```


