#include "derivatives.hpp"
#include "dynamics.hpp"
#include "rotation_types.hpp"
#include "vector.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>
#include <string>
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition))                                                          \
      throw std::runtime_error(std::string(__FILE__) + ":" +                   \
                               std::to_string(__LINE__) +                      \
                               " failed: " #condition);                        \
  } while (false)
#define CHECK_NEAR(actual, expected, tolerance)                                \
  CHECK(std::abs((actual) - (expected)) <= (tolerance))
namespace math = omni::math;
void test_dynamics() {
  const Eigen::Vector3d omega(0.2, -0.4, 0.7);
  const Eigen::Vector3d torque(0.03, -0.01, 0.02);
  Eigen::Matrix3d inertia;
  inertia << 4.0, 0.2, -0.1, 0.2, 5.0, 0.3, -0.1, 0.3, 6.0;
  const Eigen::Vector3d expected =
      inertia.ldlt().solve(torque - omega.cross(inertia * omega));
  CHECK(math::rigid_body_angular_acceleration(omega, torque, inertia)
            .isApprox(expected, 1e-14));

  const Eigen::Matrix3d spherical_inertia = 3.0 * Eigen::Matrix3d::Identity();
  const Eigen::Vector3d spherical_torque(0.3, -0.6, 0.9);
  CHECK(math::rigid_body_angular_acceleration(omega, spherical_torque,
                                              spherical_inertia)
            .isApprox(spherical_torque / 3.0, 1e-14));
  CHECK(math::rigid_body_angular_acceleration(Eigen::Vector3d(0, 0, 0.7),
                                              Eigen::Vector3d::Zero(),
                                              inertia.diagonal().asDiagonal())
            .norm() < 1e-14);

  Eigen::Matrix<double, Eigen::Dynamic, 3> axes(2, 3);
  axes << 1, 0, 0, 0, 0.6, 0.8;
  Eigen::VectorXd wheel_momentum(2);
  wheel_momentum << 0.04, -0.03;
  const Eigen::Vector3d wheel_torque(0.01, 0.02, -0.04);
  const Eigen::Vector3d total_momentum =
      inertia * omega + axes.transpose() * wheel_momentum;
  const Eigen::Vector3d wheel_expected =
      inertia.ldlt().solve(wheel_torque - omega.cross(total_momentum));
  const Eigen::Vector3d wheel_actual = math::wheel_coupled_angular_acceleration(
      omega, wheel_momentum, axes, wheel_torque, inertia);
  CHECK(wheel_actual.isApprox(wheel_expected, 1e-14));
  CHECK(!wheel_actual.isApprox(
      math::wheel_coupled_angular_acceleration(omega, Eigen::VectorXd::Zero(2),
                                               axes, wheel_torque, inertia),
      1e-8));

  Eigen::Matrix<double, Eigen::Dynamic, 3> no_axes(0, 3);
  const Eigen::Vector3d rigid =
      math::rigid_body_angular_acceleration(omega, torque, inertia);
  CHECK(math::wheel_coupled_angular_acceleration(omega, Eigen::VectorXd(0),
                                                 no_axes, torque, inertia)
            .isApprox(rigid, 1e-14));

  Eigen::Matrix<double, Eigen::Dynamic, 3> three_axes(3, 3);
  three_axes << 1, 0, 0, 0, 1, 0, 1 / std::sqrt(2.0), 0, 1 / std::sqrt(2.0);
  Eigen::Vector3d wheel_torques(0.2, -0.1, 0.05);
  const Eigen::Vector3d alpha(0.3, -0.4, 0.2);
  CHECK(math::wheel_momentum_derivative(wheel_torques, alpha, three_axes)
            .isApprox(wheel_torques - three_axes * alpha, 1e-14));
  CHECK(math::wheel_momentum_derivative(Eigen::VectorXd(0), alpha, no_axes)
            .size() == 0);
}
int main() { void test_dynamics(); }
