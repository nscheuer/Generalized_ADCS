#include "interpolation.hpp"
#include <algorithm>
#include <cmath>

namespace omni::rotations
{
Vector3 quaternion_to_rotation_vector(const QuaternionVector &quaternion)
{
    QuaternionVector unit_q = normalize_quaternion(quaternion);
    if (unit_q[0] < 0.0)
        unit_q = -unit_q;
    const double vector_norm = unit_q.tail<3>().norm();
    if (vector_norm == 0.0)
        return Vector3::Zero();
    const double angle = 2.0 * std::atan2(vector_norm, unit_q[0]);
    return unit_q.tail<3>() * (angle / vector_norm);
}

QuaternionVector slerp(const QuaternionVector &start, const QuaternionVector &finish, double fraction)
{
    QuaternionVector a = normalize_quaternion(start);
    QuaternionVector b = normalize_quaternion(finish);
    double cosine = a.dot(b);
    if (cosine < 0.0)
    {
        b = -b;
        cosine = -cosine;
    }
    cosine = std::min(cosine, 1.0);
    if (cosine > 0.9995)
        return normalize_quaternion((1.0 - fraction) * a + fraction * b);
    const double angle = std::acos(cosine);
    const double denominator = std::sin(angle);
    return normalize_quaternion((std::sin((1.0 - fraction) * angle) * a + std::sin(fraction * angle) * b) /
                                denominator);
}
} // namespace omni::rotations
