#pragma once
#include "../../environment/native_models.hpp"
#include <Eigen/Dense>
#include <memory>
#include <utility>
namespace omni::propagation {
using Vector = Eigen::VectorXd;
struct Runtime {
  double mu = 0, radius = 0, mass = 1;
  Eigen::Matrix3d inertia = Eigen::Matrix3d::Identity(),
                  inverse_inertia = Eigen::Matrix3d::Identity(),
                  inverse_bus_inertia = Eigen::Matrix3d::Identity();
  Eigen::VectorXd zonals, areas, drag_coefficients, wheel_inertias,
      wheel_commands, wheel_max_momentum, mtq_max_dipole, dipole_commands;
  Eigen::MatrixXd normals, arms, optical_coefficients, wheel_axes, mtq_axes;
  Eigen::Vector3d dipole = Eigen::Vector3d::Zero();
  bool drag_force = false, drag_torque = false, srp_force = false,
       srp_torque = false, gravity_gradient = false, residual_dipole = false,
       third_body = false, magnetic_include_static = false;
  int magnetic_kind = 0;
  std::unique_ptr<omni::environment::DE421> ephemeris;
  std::unique_ptr<omni::environment::IGRF14> igrf;
  std::unique_ptr<omni::environment::CHAOS8> chaos;
  Runtime() = default;
  void initialize_models();
};
Vector derivative(const Vector &y, double jd, const Runtime &runtime);
void project(Vector &y, const Runtime &runtime);
} // namespace omni::propagation
