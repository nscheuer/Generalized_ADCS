#include "adcs_subsystem.hpp"

#include <stdexcept>
#include <utility>

namespace omni::hardware {
ADCSSubsystem::ADCSSubsystem(
    std::vector<ReactionWheel> reaction_wheels_value,
    std::vector<Magnetorquer> magnetorquers_value,
    std::vector<Sensor> sensors_value, int N_value,
    std::shared_ptr<estimator::AttitudeEstimator> estimator_value)
    : sensors(std::move(sensors_value)), N(N_value),
      estimator(std::move(estimator_value)),
      reaction_wheels(std::move(reaction_wheels_value)),
      magnetorquers(std::move(magnetorquers_value)) {
  if (N < 1) {
    throw std::invalid_argument("N must be a positive integer");
}
}
bool ADCSSubsystem::every_step(std::uint64_t tick) const {
  if (N < 1) {
    throw std::invalid_argument("N must be a positive integer");
}
  return tick % static_cast<std::uint64_t>(N) == 0;
}
} // namespace omni::hardware
