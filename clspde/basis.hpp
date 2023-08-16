#include <iostream>
#include <vector>
#include <cmath>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <xtensor-blas/xlinalg.hpp>

class Basis {
public:
    unsigned long n;
    unsigned long n_dims;
    xt::xarray<double> steps;
    xt::xarray<double> precalculated_mults;

    Basis(unsigned long num_of_elems, xt::xarray<double> steps, unsigned long n_dims = 1) {
        
        this->n = num_of_elems;
        this->n_dims = n_dims;
        this->steps = steps;
        
        xt::xarray<double>::shape_type shape = {n_dims, n, n};
        this->precalculated_mults = xt::xarray<double>::from_shape(shape);

        // xt::xarray<double, xt::xshape<n_dims, n, n>> a();
        for (unsigned long i = 0; i < this->n_dims; i++) {
            for (unsigned long n = 0; n < this->n; n++) {
                for (unsigned long der = 0; der < this->n; der++) {
                    double prod = 1.0;
                    // xt::xarray<double> prod = {1.};
                    if (n+1 > der){
                        for (unsigned long j = n+1-der; j <= n; j++) {
                            prod *= j;
                        }
                    }
                    else{
                        prod *= 0;
                    }
                    this->precalculated_mults(i,n,der) = prod;
                }
            }
        }
    }

    xt::xarray<double> eval(xt::xarray<double> x, xt::xarray<unsigned long> derivative = xt::xarray<unsigned long>(), bool ravel = false) {
        // derivative = xt::abs(derivative);
        xt::xarray<double> result(n_dims);
        for (unsigned long i = 0; i < n_dims; i++) {
            for (unsigned long n = 0; n < this->n; n++) {
                double mult;
                try {
                    mult = this->precalculated_mults(i,n,derivative(i));
                } catch (const std::out_of_range& e) {
                    mult = 0;
                }
                
                result(i,n) = std::pow(x(i), std::max(n, derivative(i))-derivative(i)) * mult / std::pow(steps(i) / 2, derivative(i));
            }
        }
        
        if (ravel) {
            xt::xarray<double> mat_result = result(0);
            xt::xarray<double> res_i;
            for (int i = 1; i < n_dims; i++) {
                res_i = result(i);
                xt::linalg::outer(mat_result, res_i);
            } 
            return xt::ravel(mat_result);
        } else {
            return result;
        }
    }
};