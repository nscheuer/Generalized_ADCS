#pragma once
#include "_coupled_dynamics.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace omni::propagation {
inline Runtime runtime_from_python(const pybind11::object &p) {
  Runtime r;
  r.mu = p.attr("mu").cast<double>();
  r.radius = p.attr("radius").cast<double>();
  r.mass = p.attr("mass").cast<double>();
  r.inertia = p.attr("inertia").cast<Eigen::Matrix3d>();
  r.inverse_inertia = p.attr("inverse_inertia").cast<Eigen::Matrix3d>();
  r.inverse_bus_inertia = p.attr("inverse_bus_inertia").cast<Eigen::Matrix3d>();
  r.zonals = p.attr("zonals").cast<Vector>();
  r.areas = p.attr("areas").cast<Vector>();
  r.drag_coefficients = p.attr("drag_coefficients").cast<Vector>();
  r.wheel_inertias = p.attr("wheel_inertias").cast<Vector>();
  r.wheel_commands = p.attr("wheel_commands").cast<Vector>();
  r.wheel_max_momentum = p.attr("wheel_max_momentum").cast<Vector>();
  r.mtq_axes = p.attr("mtq_axes").cast<Eigen::MatrixXd>();
  r.mtq_max_dipole = p.attr("mtq_max_dipole").cast<Vector>();
  r.dipole_commands = p.attr("dipole_commands").cast<Vector>();
  r.normals = p.attr("normals").cast<Eigen::MatrixXd>();
  r.arms = p.attr("arms").cast<Eigen::MatrixXd>();
  r.optical_coefficients =
      p.attr("optical_coefficients").cast<Eigen::MatrixXd>();
  r.wheel_axes = p.attr("wheel_axes").cast<Eigen::MatrixXd>();
  r.dipole = p.attr("dipole").cast<Eigen::Vector3d>();
  r.drag_force = p.attr("drag_force").cast<bool>();
  r.drag_torque = p.attr("drag_torque").cast<bool>();
  r.srp_force = p.attr("srp_force").cast<bool>();
  r.srp_torque = p.attr("srp_torque").cast<bool>();
  r.gravity_gradient = p.attr("gravity_gradient").cast<bool>();
  r.residual_dipole = p.attr("residual_dipole").cast<bool>();
  r.third_body = p.attr("third_body").cast<bool>();
  r.magnetic_include_static = p.attr("magnetic_include_static").cast<bool>();
  r.magnetic_kind = p.attr("magnetic_kind").cast<int>();
  r.initialize_models();
  return r;
}
} // namespace omni::propagation
