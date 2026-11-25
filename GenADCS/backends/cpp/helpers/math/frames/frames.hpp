#ifndef ADCS_OMNI_FRAMES_FRAMES_HPP
#define ADCS_OMNI_FRAMES_FRAMES_HPP

#include <Eigen/Core>

namespace omni::frames {
using Vector3 = Eigen::Vector3d;
using Quaternion = Eigen::Matrix<double, 4, 1>;
using Matrix3 = Eigen::Matrix3d;

Matrix3
inertial_to_body_matrix(const Quaternion &quaternion_body_from_inertial);
Matrix3
body_to_inertial_matrix(const Quaternion &quaternion_body_from_inertial);
Vector3 inertial_to_body(const Vector3 &vector_inertial,
                         const Quaternion &quaternion_body_from_inertial);
Vector3 body_to_inertial(const Vector3 &vector_body,
                         const Quaternion &quaternion_body_from_inertial);
Vector3 transform_vector(const Vector3 &vector,
                         const Matrix3 &source_to_target);
Matrix3 transform_covariance(const Matrix3 &covariance_source,
                             const Matrix3 &source_to_target);
} // namespace omni::frames

#endif // ADCS_OMNI_FRAMES_FRAMES_HPP
