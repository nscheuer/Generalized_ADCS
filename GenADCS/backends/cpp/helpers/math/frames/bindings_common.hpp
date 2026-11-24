#pragma once

#include "frames.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;
namespace omni_frames = omni::frames;

namespace frame_bindings
{
inline py::array_t<double> vector_to_array(const omni_frames::Vector3 &vector)
{
    py::array_t<double> result(3);
    auto output = result.mutable_unchecked<1>();
    for (py::ssize_t index = 0; index < 3; ++index)
    {
        output(index) = vector[index];
    }
    return result;
}

inline py::array_t<double> matrix_to_array(const omni_frames::Matrix3 &matrix)
{
    py::array_t<double> result({3, 3});
    auto output = result.mutable_unchecked<2>();
    for (py::ssize_t row = 0; row < 3; ++row)
    {
        for (py::ssize_t column = 0; column < 3; ++column)
        {
            output(row, column) = matrix(row, column);
        }
    }
    return result;
}

inline py::array_t<double> as_array(py::handle value, const char *name, py::ssize_t rank, py::ssize_t first_extent,
                                    py::ssize_t second_extent = -1)
{
    py::array array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array)
    {
        throw py::type_error(std::string(name) + " must be convertible to a float64 array");
    }
    if ((array.ndim() != rank) || (array.shape(0) != first_extent) ||
        ((rank == 2) && (array.shape(1) != second_extent)))
    {
        throw py::value_error(std::string(name) + " has an invalid shape");
    }
    py::object numpy = py::module_::import("numpy");
    if (!numpy.attr("all")(numpy.attr("isfinite")(array)).cast<bool>())
    {
        throw py::value_error(std::string(name) + " must contain only finite values");
    }
    return py::reinterpret_borrow<py::array_t<double>>(array);
}

template <int Size> Eigen::Matrix<double, Size, 1> to_fixed_vector(py::handle value, const char *name)
{
    auto array = as_array(value, name, 1, Size);
    auto input = array.unchecked<1>();
    Eigen::Matrix<double, Size, 1> result;
    for (int index = 0; index < Size; ++index)
    {
        result[index] = input(index);
    }
    return result;
}

inline omni_frames::Matrix3 to_matrix(py::handle value, const char *name)
{
    auto array = as_array(value, name, 2, 3, 3);
    auto input = array.unchecked<2>();
    omni_frames::Matrix3 result;
    for (py::ssize_t row = 0; row < 3; ++row)
    {
        for (py::ssize_t column = 0; column < 3; ++column)
        {
            result(row, column) = input(row, column);
        }
    }
    return result;
}

} // namespace frame_bindings
