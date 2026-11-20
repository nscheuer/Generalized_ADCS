#ifndef ADCS_OMNI_HELPERS_DYNAMICS_HPP
#define ADCS_OMNI_HELPERS_DYNAMICS_HPP

#include <Eigen/Core>

namespace omni::math {
Eigen::Vector3d rigid_body_angular_acceleration(const Eigen::Vector3d& angular_velocity,
                                               const Eigen::Vector3d& torque,
                                               const Eigen::Matrix3d& inertia);
Eigen::Vector3d wheel_coupled_angular_acceleration(const Eigen::Vector3d& angular_velocity,
                                                  const Eigen::VectorXd& wheel_momentum,
                                                  const Eigen::Matrix<double, Eigen::Dynamic, 3>& wheel_axes,
                                                  const Eigen::Vector3d& torque,
                                                  const Eigen::Matrix3d& bus_inertia);
Eigen::VectorXd wheel_momentum_derivative(const Eigen::VectorXd& wheel_torque,
                                          const Eigen::Vector3d& angular_acceleration,
                                          const Eigen::Matrix<double, Eigen::Dynamic, 3>& wheel_axes);
}

#endif  // ADCS_OMNI_HELPERS_DYNAMICS_HPP
