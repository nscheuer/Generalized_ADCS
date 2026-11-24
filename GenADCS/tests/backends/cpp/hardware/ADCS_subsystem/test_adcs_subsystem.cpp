#include "adcs_subsystem.hpp"
#include "check.hpp"
#include <stdexcept>

using omni::hardware::ADCSSubsystem;
using omni::hardware::Magnetorquer;
using omni::hardware::ReactionWheel;

int main()
{
    const ADCSSubsystem scheduled({}, {}, {}, 3);
    CHECK(scheduled.N == 3);
    CHECK(scheduled.every_step(0));
    CHECK(!scheduled.every_step(1));
    CHECK(scheduled.every_step(3));
    bool rejected = false;
    try { ADCSSubsystem invalid({}, {}, {}, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    const ADCSSubsystem empty;
    CHECK(empty.reaction_wheels.empty());
    CHECK(empty.magnetorquers.empty());

    const ReactionWheel wheel({{1.0, 2.0, 3.0}}, {{0.0, 0.0, 1.0}}, 0.05, 0.15, 3.0);
    const Magnetorquer torquer({{0.0, 1.0, 0.0}}, {{1.0, 0.0, 0.0}}, 0.25);
    const ADCSSubsystem populated({wheel}, {torquer});
    CHECK(populated.reaction_wheels.size() == 1);
    CHECK(populated.magnetorquers.size() == 1);
    CHECK((populated.reaction_wheels.front().position == ReactionWheel::Vector3{{1.0, 2.0, 3.0}}));
    CHECK((populated.reaction_wheels.front().orientation == ReactionWheel::Vector3{{0.0, 0.0, 1.0}}));
    CHECK(populated.reaction_wheels.front().rotational_inertia == 0.05);
    CHECK(populated.reaction_wheels.front().max_torque == 0.15);
    CHECK(populated.reaction_wheels.front().max_momentum == 3.0);
    CHECK((populated.magnetorquers.front().position == Magnetorquer::V{{0.0, 1.0, 0.0}}));
    CHECK((populated.magnetorquers.front().orientation == Magnetorquer::V{{1.0, 0.0, 0.0}}));
    CHECK(populated.magnetorquers.front().max_dipole == 0.25);

    ADCSSubsystem mutable_subsystem;
    mutable_subsystem.reaction_wheels.push_back(wheel);
    mutable_subsystem.magnetorquers.push_back(torquer);
    mutable_subsystem.reaction_wheels.front().max_torque = 0.2;
    CHECK(mutable_subsystem.reaction_wheels.front().max_torque == 0.2);
    CHECK(empty.reaction_wheels.empty());
    CHECK(empty.magnetorquers.empty());
}
