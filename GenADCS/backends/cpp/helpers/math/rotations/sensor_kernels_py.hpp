#pragma once
#include "sensor_kernels.hpp"
#include <pybind11/numpy.h>
namespace omni::hardware::sensors::kernels {
V read_vector3(pybind11::handle value, const char *name);
Q read_quaternion(pybind11::handle value, const char *name);
M read_matrix(pybind11::handle value, const char *name);
pybind11::array_t<double> array_vector(const V &v);
pybind11::array_t<double> array_matrix(const M &m);
} // namespace omni::hardware::sensors::kernels
