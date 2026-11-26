#include "propagation/attitude_propagators/rk4_attitude_propagator.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
  using namespace omni::propagation;
  Runtime runtime;
  runtime.inertia = Eigen::Matrix3d::Identity();
  runtime.inverse_inertia = Eigen::Matrix3d::Identity();
  runtime.inverse_bus_inertia = Eigen::Matrix3d::Identity();
  runtime.wheel_commands.resize(0);
  Vector state = Vector::Zero(13);
  state[0] = 1.0;
  state[6] = 0.1;
  state[7] = 7.0e6;
  state[11] = 7500.0;

  const Vector result = rk4_attitude_step(state, 2451545.0, 1.0, runtime);
  const double angle = 0.05;
  if (std::abs(result[0] - std::cos(angle)) > 1e-8 ||
      std::abs(result[3] + std::sin(angle)) > 1e-8 ||
      std::abs(result[6] - 0.1) > 1e-12 ||
      (result.segment<6>(7) - state.segment<6>(7)).norm() != 0.0) {
    std::cerr << "torque-free RK4 attitude step mismatch\n";
    return 1;
  }

  const auto trajectory =
      integrate_attitude_rk4(state, 2451545.0, 0.25, 1.1, runtime, 3);
  if (trajectory.times.size() != 3 || trajectory.times[0] != 0.0 ||
      std::abs(trajectory.times[1] - 0.75) > 1e-12 ||
      std::abs(trajectory.times[2] - 1.1) > 1e-12 ||
      std::abs(trajectory.states.back()[0] - std::cos(0.055)) > 1e-9 ||
      std::abs(trajectory.states.back()[3] + std::sin(0.055)) > 1e-9) {
    std::cerr << "attitude RK4 integration mismatch\n";
    return 1;
  }

  try {
    (void)rk4_attitude_step(state, 2451545.0, 0.0, runtime);
    std::cerr << "zero step duration was accepted\n";
    return 1;
  } catch (const std::invalid_argument &) {
  }
  return 0;
}
