#include "reaction_wheel.hpp"

#include <stdexcept>
#include <string>

using omni::hardware::ReactionWheel;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition))                                                          \
      throw std::runtime_error(std::string(__FILE__) + ":" +                   \
                               std::to_string(__LINE__) +                      \
                               " failed: " #condition);                        \
  } while (false)

int main() {
  const ReactionWheel wheel({{1.0, 2.0, 3.0}}, {{0.0, 0.0, 1.0}}, 0.04, 0.2,
                            5.0);
  CHECK((wheel.position == ReactionWheel::Vector3{{1.0, 2.0, 3.0}}));
  CHECK((wheel.orientation == ReactionWheel::Vector3{{0.0, 0.0, 1.0}}));
  CHECK(wheel.rotational_inertia == 0.04);
  CHECK(wheel.max_torque == 0.2);
  CHECK(wheel.max_momentum == 5.0);
  CHECK((wheel.misalignment == ReactionWheel::Vector3{{0.0, 0.0, 0.0}}));
  CHECK(wheel.noise == 0.0);
  CHECK(wheel.bias == 0.0);

  const ReactionWheel misaligned({{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, 0.03,
                                 0.1, 2.0, {{0.01, 0.0, -0.01}}, 0.02, -0.01);
  CHECK(
      (misaligned.misalignment == ReactionWheel::Vector3{{0.01, 0.0, -0.01}}));
  CHECK(misaligned.noise == 0.02);
  CHECK(misaligned.bias == -0.01);
}
