#include "dynamics.hpp"

#include <Eigen/Geometry>
#include <Eigen/LU>

#include <stdexcept>

namespace omni::math {
namespace {
Eigen::Vector3d solve_inertia(const Eigen::Matrix3d &inertia,
                              const Eigen::Vector3d &right_hand_side) {
  if (!inertia.allFinite() || !right_hand_side.allFinite()) {
    throw std::invalid_argument("inertia and right-hand side must be finite");
  }
  Eigen::FullPivLU<Eigen::Matrix3d> decomposition(inertia);
  if (!decomposition.isInvertible()) {
    throw std::invalid_argument("inertia matrix must be invertible");
}
  return decomposition.solve(right_hand_side);
}
} // namespace

Eigen::Vector3d
rigid_body_angular_acceleration(const Eigen::Vector3d &angular_velocity,
                                const Eigen::Vector3d &torque,
                                const Eigen::Matrix3d &inertia) {
  const Eigen::Vector3d angular_momentum = inertia * angular_velocity;
  return solve_inertia(inertia,
                       torque - angular_velocity.cross(angular_momentum));
}

Eigen::Vector3d wheel_coupled_angular_acceleration(
    const Eigen::Vector3d &angular_velocity,
    const Eigen::VectorXd &wheel_momentum,
    const Eigen::Matrix<double, Eigen::Dynamic, 3> &wheel_axes,
    const Eigen::Vector3d &torque, const Eigen::Matrix3d &bus_inertia) {
  if (wheel_axes.rows() != wheel_momentum.size()) {
    throw std::invalid_argument(
        "wheel_axes row count must match wheel_momentum size");
  }
  const Eigen::Vector3d total_momentum =
      bus_inertia * angular_velocity + wheel_axes.transpose() * wheel_momentum;
  return solve_inertia(bus_inertia,
                       torque - angular_velocity.cross(total_momentum));
}

Eigen::VectorXd wheel_momentum_derivative(
    const Eigen::VectorXd &wheel_torque,
    const Eigen::Vector3d &angular_acceleration,
    const Eigen::Matrix<double, Eigen::Dynamic, 3> &wheel_axes) {
  if (wheel_axes.rows() != wheel_torque.size()) {
    throw std::invalid_argument(
        "wheel_axes row count must match wheel_torque size");
  }
  return wheel_torque - wheel_axes * angular_acceleration;
}
} // namespace omni::math
