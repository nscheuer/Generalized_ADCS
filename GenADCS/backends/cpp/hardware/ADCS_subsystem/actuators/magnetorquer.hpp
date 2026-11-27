#ifndef ADCS_OMNI_HARDWARE_MAGNETORQUER_HPP
#define ADCS_OMNI_HARDWARE_MAGNETORQUER_HPP

#include <array>

namespace omni::hardware {
struct Magnetorquer {
  using V = std::array<double, 3>;

  Magnetorquer(V position_value, V orientation_value, double max_dipole_value,
               V misalignment_value = {{0.0, 0.0, 0.0}},
               double noise_value = 0.0, double bias_value = 0.0);

  V position;
  V orientation;
  double max_dipole;
  V misalignment;
  double noise;
  double bias;
};
} // namespace omni::hardware

#endif
