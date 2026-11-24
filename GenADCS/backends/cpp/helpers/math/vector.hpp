#ifndef ADCS_OMNI_HELPERS_VECTOR_HPP
#define ADCS_OMNI_HELPERS_VECTOR_HPP

#include <Eigen/Core>

namespace omni::math
{
Eigen::Vector3d cross(const Eigen::Vector3d &left, const Eigen::Vector3d &right);
double norm(const Eigen::VectorXd &vector);
Eigen::VectorXd normalize(const Eigen::VectorXd &vector);
Eigen::VectorXd unit(const Eigen::VectorXd &vector);
Eigen::Matrix3d skew(const Eigen::Vector3d &vector);
} // namespace omni::math

#endif // ADCS_OMNI_HELPERS_VECTOR_HPP
