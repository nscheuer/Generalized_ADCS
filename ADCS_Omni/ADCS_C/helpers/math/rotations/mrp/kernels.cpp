#include "kernels.hpp"
#include "../quaternion/algebra.hpp"

namespace omni::rotations {
QuaternionVector mrp_to_quaternion_parameters(const Vector3& parameters) {
    const double squared_norm = parameters.squaredNorm();
    const double denominator = 1.0 + squared_norm;
    QuaternionVector quaternion;
    quaternion[0] = (1.0 - squared_norm) / denominator;
    quaternion.tail<3>() = 2.0 * parameters / denominator;
    return quaternion;
}

Vector3 quaternion_to_mrp_parameters(const QuaternionVector& quaternion) {
    QuaternionVector q = normalize_quaternion(quaternion);
    if (q[0] < 0.0) q = -q;
    Vector3 parameters = q.tail<3>() / (1.0 + q[0]);
    const double squared_norm = parameters.squaredNorm();
    if (squared_norm > 1.0) parameters = -parameters / squared_norm;
    return parameters;
}

Vector3 mrp_shadow_set(const Vector3& parameters) {
    const double squared_norm = parameters.squaredNorm();
    if (squared_norm == 0.0) throw std::invalid_argument("the identity MRP has no finite shadow set");
    return -parameters / squared_norm;
}
}  // namespace omni::rotations
