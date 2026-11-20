#pragma once
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <memory>
#include "../../environment/native_models.hpp"
namespace omni::propagation {
using Vector=Eigen::VectorXd;
struct Runtime {
 double mu,radius,mass;
 Eigen::Matrix3d inertia,inverse_inertia,inverse_bus_inertia;
 Eigen::VectorXd zonals,areas,drag_coefficients,wheel_inertias,wheel_commands,wheel_max_momentum;
 Eigen::MatrixXd normals,arms,optical_coefficients,wheel_axes;
 Eigen::Vector3d dipole;
 bool drag_force,drag_torque,srp_force,srp_torque,gravity_gradient,residual_dipole,third_body,magnetic_include_static;
 int magnetic_kind;
 std::unique_ptr<omni::environment::DE421> ephemeris;
 std::unique_ptr<omni::environment::IGRF14> igrf;
 std::unique_ptr<omni::environment::CHAOS8> chaos;
 explicit Runtime(const pybind11::object& source);
};
struct RuntimeScope {
 Runtime runtime;
 Runtime* previous;
 explicit RuntimeScope(const pybind11::object& source);
 ~RuntimeScope();
};
Vector derivative(const Vector& y,double jd,const pybind11::object& runtime);
void project(Vector& y,const pybind11::object& runtime);
}
