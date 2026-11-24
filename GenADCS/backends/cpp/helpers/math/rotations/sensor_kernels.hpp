#ifndef GENADCS_MATH_ROTATIONS_SENSOR_KERNELS_HPP
#define GENADCS_MATH_ROTATIONS_SENSOR_KERNELS_HPP
#include <pybind11/numpy.h>
#include <array>
namespace omni::hardware::sensors::kernels {
using V = std::array<double, 3>;
using Q = std::array<double, 4>;
using M = std::array<std::array<double, 3>, 3>;
V read_vector3(pybind11::handle value, const char* name);
Q read_quaternion(pybind11::handle value, const char* name);
M read_matrix(pybind11::handle value, const char* name);
Q normalize(Q q);
Q multiply(const Q& a, const Q& b);
M identity();
M quaternion_matrix(const Q& q);
V multiply(const M& a, const V& v);
M multiply(const M& a, const M& b);
M transpose(const M& a);
M add(const M& a, const M& b);
M scale(double s, const M& a);
M skew(const V& v);
M outer(const V& a, const V& b);
M cholesky(const M& a);
M inverse_from_cholesky(const M& l);
pybind11::array_t<double> array_vector(const V& v);
pybind11::array_t<double> array_matrix(const M& m);
Q rotation_vector_quaternion(const V& v);
V rotate_inertial_to_sensor(const V& field, const Q& attitude, const Q& orientation, const Q& misalignment);
V rotate_body_to_sensor(const V& body, const Q& orientation, const Q& misalignment);
M attitude_vector_jacobian_quaternion(const V& vector, const Q& delta);
M attitude_vector_jacobian_mrp(const V& vector, const V& delta);
M attitude_vector_jacobian_angle(const V& vector, const V& delta);

}
#endif
