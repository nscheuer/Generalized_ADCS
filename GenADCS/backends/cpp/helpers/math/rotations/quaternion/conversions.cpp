#include "conversions.hpp"
#include "../../vector.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

namespace omni::rotations {
namespace {
void require_finite(const Vector3& value, const char* description) {
    if (!value.allFinite()) throw std::invalid_argument(std::string(description) + " must be finite");
}
}
Matrix3 quaternion_to_dcm(const QuaternionVector& q) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    Matrix3 result;
    result << w*w + x*x - y*y - z*z, 2.0*(x*y-w*z), 2.0*(x*z+w*y),
              2.0*(x*y+w*z), w*w-x*x+y*y-z*z, 2.0*(y*z-w*x),
              2.0*(x*z-w*y), 2.0*(y*z+w*x), w*w-x*x-y*y+z*z;
    return result;
}

QuaternionVector dcm_to_quaternion(const Matrix3& matrix) {
    QuaternionVector q;
    const double trace = matrix.trace();
    if (trace > 0.0) {
        const double scale = 2.0 * std::sqrt(trace + 1.0);
        q[0] = 0.25 * scale;
        q[1] = (matrix(2, 1) - matrix(1, 2)) / scale;
        q[2] = (matrix(0, 2) - matrix(2, 0)) / scale;
        q[3] = (matrix(1, 0) - matrix(0, 1)) / scale;
    } else if ((matrix(0, 0) > matrix(1, 1)) && (matrix(0, 0) > matrix(2, 2))) {
        const double scale = 2.0 * std::sqrt(1.0 + matrix(0, 0) - matrix(1, 1) - matrix(2, 2));
        q[0] = (matrix(2, 1) - matrix(1, 2)) / scale;
        q[1] = 0.25 * scale;
        q[2] = (matrix(0, 1) + matrix(1, 0)) / scale;
        q[3] = (matrix(0, 2) + matrix(2, 0)) / scale;
    } else if (matrix(1, 1) > matrix(2, 2)) {
        const double scale = 2.0 * std::sqrt(1.0 + matrix(1, 1) - matrix(0, 0) - matrix(2, 2));
        q[0] = (matrix(0, 2) - matrix(2, 0)) / scale;
        q[1] = (matrix(0, 1) + matrix(1, 0)) / scale;
        q[2] = 0.25 * scale;
        q[3] = (matrix(1, 2) + matrix(2, 1)) / scale;
    } else {
        const double scale = 2.0 * std::sqrt(1.0 + matrix(2, 2) - matrix(0, 0) - matrix(1, 1));
        q[0] = (matrix(1, 0) - matrix(0, 1)) / scale;
        q[1] = (matrix(0, 2) + matrix(2, 0)) / scale;
        q[2] = (matrix(1, 2) + matrix(2, 1)) / scale;
        q[3] = 0.25 * scale;
    }
    return normalize_quaternion(q);
}

QuaternionVector axis_angle_to_quaternion(const Vector3& axis, double angle) {
    require_finite(axis, "axis");
    if (!std::isfinite(angle)) throw std::invalid_argument("angle must be finite");
    const double axis_norm = axis.norm();
    if (axis_norm == 0.0) throw std::invalid_argument("axis must be nonzero");
    const double half_angle = 0.5 * angle;
    QuaternionVector result;
    result[0] = std::cos(half_angle);
    result.tail<3>() = axis * (std::sin(half_angle) / axis_norm);
    return result;
}

QuaternionVector rotation_vector_to_quaternion(const Vector3& rotation_vector) {
    require_finite(rotation_vector, "rotation_vector");
    const double angle = rotation_vector.norm();
    if (angle == 0.0) return QuaternionVector(1.0, 0.0, 0.0, 0.0);
    const double half_angle = 0.5 * angle;
    QuaternionVector result;
    result[0] = std::cos(half_angle);
    result.tail<3>() = rotation_vector * (std::sin(half_angle) / angle);
    return result;
}
}  // namespace omni::rotations
