#include "vector.hpp"

#include <Eigen/Geometry>
#include <stdexcept>

namespace omni::math
{
Eigen::Vector3d cross(const Eigen::Vector3d &left, const Eigen::Vector3d &right)
{
    return left.cross(right);
}

double norm(const Eigen::VectorXd &vector)
{
    return vector.norm();
}

Eigen::VectorXd normalize(const Eigen::VectorXd &vector)
{
    const double magnitude = vector.norm();
    if (magnitude == 0.0)
        return Eigen::VectorXd::Zero(vector.size());
    return vector / magnitude;
}

Eigen::VectorXd unit(const Eigen::VectorXd &vector)
{
    const double magnitude = vector.norm();
    if (magnitude == 0.0)
        throw std::invalid_argument("cannot normalize a zero vector");
    return vector / magnitude;
}

Eigen::Matrix3d skew(const Eigen::Vector3d &vector)
{
    Eigen::Matrix3d result;
    result << 0.0, -vector[2], vector[1], vector[2], 0.0, -vector[0], -vector[1], vector[0], 0.0;
    return result;
}
} // namespace omni::math
