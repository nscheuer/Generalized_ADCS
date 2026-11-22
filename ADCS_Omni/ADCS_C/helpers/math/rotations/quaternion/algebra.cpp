#include "algebra.hpp"
#include "../../vector.hpp"
#include <Eigen/Geometry>
#include <stdexcept>
#include <string>

namespace omni::rotations {
namespace {
void require_finite(const QuaternionVector& value, const char* description) {
    if (!value.allFinite()) throw std::invalid_argument(std::string(description) + " must be finite");
}
void require_finite(const Vector3& value, const char* description) {
    if (!value.allFinite()) throw std::invalid_argument(std::string(description) + " must be finite");
}
}
QuaternionVector normalize_quaternion(const QuaternionVector& value) {
    require_finite(value, "quaternion");
    const double magnitude = value.norm();
    if (magnitude == 0.0) throw std::invalid_argument("quaternion must have nonzero norm");
    return value / magnitude;
}

double quaternion_norm(const QuaternionVector& value) { return value.norm(); }

QuaternionVector jpl_to_hamilton(const QuaternionVector& value) {
    QuaternionVector result;
    result << value[3], -value[0], -value[1], -value[2];
    return result;
}

QuaternionVector hamilton_to_jpl(const QuaternionVector& value) {
    QuaternionVector result;
    result << -value[1], -value[2], -value[3], value[0];
    return result;
}

QuaternionVector conjugate(const QuaternionVector& value) {
    QuaternionVector result = value;
    result.tail<3>() *= -1.0;
    return result;
}

QuaternionVector inverse(const QuaternionVector& value) {
    const double squared_norm = value.squaredNorm();
    if (squared_norm == 0.0) throw std::invalid_argument("zero quaternion has no inverse");
    return conjugate(value) / squared_norm;
}

QuaternionVector multiply(const QuaternionVector& p, const QuaternionVector& q) {
    QuaternionVector result;
    result[0] = p[0] * q[0] - p.tail<3>().dot(q.tail<3>());
    result[1] = p[0] * q[1] + q[0] * p[1] + p[2] * q[3] - p[3] * q[2];
    result[2] = p[0] * q[2] + q[0] * p[2] + p[3] * q[1] - p[1] * q[3];
    result[3] = p[0] * q[3] + q[0] * p[3] + p[1] * q[2] - p[2] * q[1];
    return result;
}

QuaternionVector relative_quaternion(const QuaternionVector& reference, const QuaternionVector& target) {
    QuaternionVector relative = normalize_quaternion(multiply(target, inverse(reference)));
    if (relative[0] < 0.0) relative = -relative;
    return relative;
}

Vector3 rotate_vector(const QuaternionVector& quaternion, const Vector3& vector) {
    require_finite(quaternion, "quaternion");
    require_finite(vector, "vector");
    const Vector3 qv = quaternion.tail<3>();
    return vector + 2.0 * (quaternion[0] * qv.cross(vector) + qv.cross(qv.cross(vector)));
}
}  // namespace omni::rotations
