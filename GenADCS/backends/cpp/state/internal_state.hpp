#ifndef ADCS_OMNI_INTERNAL_STATE_HPP
#define ADCS_OMNI_INTERNAL_STATE_HPP

#include <Eigen/Core>

namespace omni {
struct InternalState {
  Eigen::VectorXd wheel_momentum;

  explicit InternalState(const Eigen::VectorXd &momentum = Eigen::VectorXd{});
  virtual ~InternalState() noexcept = default;
};

struct WheelMomentumState final : InternalState {
  using InternalState::InternalState;
};
} // namespace omni

#endif // ADCS_OMNI_INTERNAL_STATE_HPP
