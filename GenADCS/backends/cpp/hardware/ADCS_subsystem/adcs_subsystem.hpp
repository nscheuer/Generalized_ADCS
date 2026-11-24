#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_HPP

#include "magnetorquer.hpp"
#include "reaction_wheel.hpp"

#include <vector>

namespace omni::hardware
{
struct ADCSSubsystem
{
    ADCSSubsystem() = default;
    ADCSSubsystem(std::vector<ReactionWheel> reaction_wheels_value, std::vector<Magnetorquer> magnetorquers_value);

    std::vector<ReactionWheel> reaction_wheels;
    std::vector<Magnetorquer> magnetorquers;
};
} // namespace omni::hardware

#endif
