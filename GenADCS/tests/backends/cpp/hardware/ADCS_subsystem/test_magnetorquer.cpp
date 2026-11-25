#include "magnetorquer.hpp"

#include <stdexcept>
#include <string>

using omni::hardware::Magnetorquer;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition))                                                          \
      throw std::runtime_error(std::string(__FILE__) + ":" +                   \
                               std::to_string(__LINE__) +                      \
                               " failed: " #condition);                        \
  } while (false)

int main() {
  const Magnetorquer::V position{{1.0, 2.0, 3.0}};
  const Magnetorquer::V orientation{{0.0, 0.0, 1.0}};
  Magnetorquer torquer(position, orientation, 0.25);

  CHECK(torquer.position == position);
  CHECK(torquer.orientation == orientation);
  CHECK(torquer.max_dipole == 0.25);
  CHECK((torquer.misalignment == Magnetorquer::V{{0.0, 0.0, 0.0}}));

  torquer.position = {{4.0, 5.0, 6.0}};
  torquer.max_dipole = 0.5;
  torquer.misalignment = {{0.1, 0.0, -0.1}};
  CHECK((torquer.position == Magnetorquer::V{{4.0, 5.0, 6.0}}));
  CHECK(torquer.max_dipole == 0.5);
  CHECK((torquer.misalignment == Magnetorquer::V{{0.1, 0.0, -0.1}}));
}
