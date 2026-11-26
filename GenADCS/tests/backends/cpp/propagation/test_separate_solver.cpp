#include "propagation/attitude_propagators/separate_solver.hpp"

#include <cmath>
#include <stdexcept>

int main() {
  using namespace omni::propagation;
  constexpr double mu = 3.986004418e14;
  Eigen::VectorXd initial = Eigen::VectorXd::Zero(13);
  initial[0] = 1.0;
  initial[6] = 0.1;
  initial[7] = 7e6;
  initial[11] = 7500.0;
  Runtime runtime;
  runtime.inertia = Eigen::Matrix3d::Identity();
  runtime.inverse_inertia = Eigen::Matrix3d::Identity();
  runtime.inverse_bus_inertia = Eigen::Matrix3d::Identity();
  runtime.wheel_commands.resize(0);
  const auto rk4 = integrate_separate_rk4(
      initial, mu, 2451545.0, 0.25, 1.1, 3, runtime);
  if (rk4.first.size() != 3 ||
      std::abs(rk4.first.back() - 1.1) > 1e-12 ||
      std::abs(rk4.second(2, 0) - std::cos(0.055)) > 1e-9 ||
      std::abs(rk4.second(2, 3) + std::sin(0.055)) > 1e-9 ||
      rk4.second(2, 8) <= 0.0) {
    throw std::runtime_error("native separate RK4 integration failed");
  }
  const auto reference = integrate_separate_reference(
      initial, mu, 2451545.0, 0.25, 0.5, 1, "nadir");
  if (reference.first.size() != 3 ||
      std::abs(reference.second.row(0).segment<4>(0).norm() - 1.0) > 1e-12 ||
      reference.second(2, 8) <= 0.0) {
    throw std::runtime_error("native separate reference integration failed");
  }
}
