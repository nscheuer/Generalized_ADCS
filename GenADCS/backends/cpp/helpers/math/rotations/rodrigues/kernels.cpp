#include "kernels.hpp"
#include "../quaternion/algebra.hpp"

namespace omni::rotations {
QuaternionVector rodrigues_to_quaternion_parameters(const Vector3 &parameters) {
  const double denominator = std::sqrt(1.0 + parameters.squaredNorm());
  QuaternionVector quaternion;
  quaternion[0] = 1.0 / denominator;
  quaternion.tail<3>() = parameters / denominator;
  return quaternion;
}

Vector3 quaternion_to_rodrigues_parameters(const QuaternionVector &quaternion) {
  QuaternionVector q = normalize_quaternion(quaternion);
  if (q[0] < 0.0) {
    q = -q;
}
  if (std::abs(q[0]) <= 1e-12) {
    throw std::invalid_argument(
        "Rodrigues parameters are singular at a 180-degree rotation");
}
  return q.tail<3>() / q[0];
}
} // namespace omni::rotations
