#ifndef ADCS_OMNI_HELPERS_DERIVATIVES_HPP
#define ADCS_OMNI_HELPERS_DERIVATIVES_HPP

#include <Eigen/Core>

namespace omni::math {
Eigen::Vector4d quaternion_derivative(const Eigen::Vector4d &q,
                                      const Eigen::Vector3d &omega);
Eigen::Matrix<double, 4, 3> quaternion_rate_matrix(const Eigen::Vector4d &q);
Eigen::Vector3d
angular_velocity_from_quaternion_derivative(const Eigen::Vector4d &q,
                                            const Eigen::Vector4d &qdot);
Eigen::Matrix4d
quaternion_normalization_jacobian(const Eigen::Vector4d &quaternion);
Eigen::Matrix<double, 3, 4>
rotated_vector_quaternion_jacobian(const Eigen::Vector4d &q,
                                   const Eigen::Vector3d &vector);
Eigen::Vector3d quaternion_to_euler321(const Eigen::Vector4d &q);
} // namespace omni::math

#endif // ADCS_OMNI_HELPERS_DERIVATIVES_HPP
