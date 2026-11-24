#include "sensor_kernels_py.hpp"

#include <stdexcept>
#include <string>

namespace py = pybind11;
namespace omni::hardware::sensors::kernels
{
namespace
{
template <std::size_t N> std::array<double, N> read_vector(py::handle value, const char *name)
{
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array || array.ndim() != 1 || array.shape(0) != static_cast<py::ssize_t>(N))
        throw py::value_error(std::string(name) + " has an invalid shape");
    std::array<double, N> result{};
    auto input = array.unchecked<1>();
    for (std::size_t i = 0; i < N; ++i)
        result[i] = input(static_cast<py::ssize_t>(i));
    return result;
}
} // namespace
M read_matrix(py::handle value, const char *name)
{
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array || array.ndim() != 2 || array.shape(0) != 3 || array.shape(1) != 3)
        throw py::value_error(std::string(name) + " must be a 3x3 matrix");
    M result{};
    auto input = array.unchecked<2>();
    for (py::ssize_t i = 0; i < 3; ++i)
        for (py::ssize_t j = 0; j < 3; ++j)
            result[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = input(i, j);
    return result;
}
V read_vector3(py::handle value, const char *name)
{
    return read_vector<3>(value, name);
}
Q read_quaternion(py::handle value, const char *name)
{
    return read_vector<4>(value, name);
}
py::array_t<double> array_vector(const V &v)
{
    py::array_t<double> a(3);
    auto out = a.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < 3; ++i)
        out(i) = v[static_cast<std::size_t>(i)];
    return a;
}
py::array_t<double> array_matrix(const M &m)
{
    py::array_t<double> a({3, 3});
    auto out = a.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < 3; ++i)
        for (py::ssize_t j = 0; j < 3; ++j)
            out(i, j) = m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    return a;
}
} // namespace omni::hardware::sensors::kernels
