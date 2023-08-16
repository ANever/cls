#include <iostream>
#include <vector>
#include <cmath>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
     

int main() {
    std::vector<int> steps(5);
    std::vector<size_t> shape = { 3, 2, 4 };
    xt::xarray<double> a(shape);
    return 0;
}