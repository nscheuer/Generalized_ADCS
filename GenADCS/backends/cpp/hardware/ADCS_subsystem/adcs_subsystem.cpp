#include "adcs_subsystem.hpp"

#include <utility>

namespace omni::hardware {
ADCSSubsystem::ADCSSubsystem(std::vector<ReactionWheel> reaction_wheels_value,
                             std::vector<Magnetorquer> magnetorquers_value)
    : reaction_wheels(std::move(reaction_wheels_value)),
      magnetorquers(std::move(magnetorquers_value)) {}
}
