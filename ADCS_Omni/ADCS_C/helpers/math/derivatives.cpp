#include "derivatives.hpp"

#include "vector.hpp"

#include <algorithm>
#include <cmath>
#include <Eigen/Geometry>
#include <stdexcept>

namespace omni::math {
Eigen::Vector4d quaternion_derivative(const Eigen::Vector4d& q, const Eigen::Vector3d& omega) {
    Eigen::Vector4d result;
    result << q.tail<3>().dot(omega),
        -(q[0] * omega[0] + omega[1] * q[3] - omega[2] * q[2]),
        -(q[0] * omega[1] + omega[2] * q[1] - omega[0] * q[3]),
        -(q[0] * omega[2] + omega[0] * q[2] - omega[1] * q[1]);
    return 0.5 * result;
}

Eigen::Matrix<double, 4, 3> quaternion_rate_matrix(const Eigen::Vector4d& q) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    Eigen::Matrix<double, 4, 3> result;
    result << x, y, z, -w, -z, y, z, -w, -x, -y, x, -w;
    return 0.5 * result;
}

Eigen::Vector3d angular_velocity_from_quaternion_derivative(const Eigen::Vector4d& q,
                                                             const Eigen::Vector4d& qdot) {
    Eigen::Vector3d result;
    result << q[1] * qdot[0] - q[0] * qdot[1] + q[3] * qdot[2] - q[2] * qdot[3],
        q[2] * qdot[0] - q[3] * qdot[1] - q[0] * qdot[2] + q[1] * qdot[3],
        q[3] * qdot[0] + q[2] * qdot[1] - q[1] * qdot[2] - q[0] * qdot[3];
    return 2.0 * result;
}

Eigen::Matrix4d quaternion_normalization_jacobian(const Eigen::Vector4d& quaternion) {
    const double magnitude = quaternion.norm();
    if (magnitude == 0.0) throw std::invalid_argument("quaternion must have nonzero norm");
    return Eigen::Matrix4d::Identity() / magnitude
         - (quaternion * quaternion.transpose()) / std::pow(magnitude, 3.0);
}

Eigen::Matrix<double, 3, 4> rotated_vector_quaternion_jacobian(const Eigen::Vector4d& q,
                                                               const Eigen::Vector3d& vector) {
    const Eigen::Vector3d qv = q.tail<3>();
    Eigen::Matrix<double, 3, 4> result;
    result.col(0) = 2.0 * qv.cross(vector);
    result.block<3, 3>(0, 1) = 2.0 * (qv.dot(vector) * Eigen::Matrix3d::Identity()
        + qv * vector.transpose() - 2.0 * vector * qv.transpose() - q[0] * skew(vector));
    return result;
}

Eigen::Vector3d quaternion_to_euler321(const Eigen::Vector4d& q) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
    const double sine_pitch = 2.0 * (w * y - z * x);
    const double pitch = std::asin(std::clamp(sine_pitch, -1.0, 1.0));
    const double yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
    return Eigen::Vector3d(roll, pitch, yaw);
}
}
