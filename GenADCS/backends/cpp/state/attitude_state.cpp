#include "attitude_state.hpp"

#include "state_validation.hpp"

#include <stdexcept>

namespace omni {
AttitudeState::AttitudeState(const Quaternion &orientation_value,
                             const Vector3 &rates_value)
    : orientation(orientation_value), rates(rates_value) {
  detail::require_finite(orientation, "orientation");
  detail::require_finite(rates, "rates");
  if (orientation.squaredNorm() == 0.0) {
    throw std::invalid_argument(
        "orientation quaternion must be finite and nonzero");
  }
}

Quaternion AttitudeState::quaternion_array() const {
  const double norm = orientation.norm();
  if ((norm == 0.0) || !std::isfinite(norm)) {
    throw std::invalid_argument(
        "orientation quaternion must be finite and nonzero");
  }
  return orientation / norm;
}
} // namespace omni
