#ifndef ADCS_OMNI_HARDWARE_SATELLITE_PANEL_HPP
#define ADCS_OMNI_HARDWARE_SATELLITE_PANEL_HPP

#include <array>

namespace omni::hardware {
struct SatellitePanel {
  SatellitePanel(std::array<double, 3> position_value,
                 std::array<double, 3> orientation_value,
                 std::array<double, 2> dimensions_value,
                 double eta_s_value = 0.0, double eta_d_value = 0.0,
                 double eta_a_value = 1.0, double cd_value = 2.2);

  std::array<double, 3> position;
  std::array<double, 3> orientation;
  std::array<double, 2> dimensions;
  double eta_s;
  double eta_d;
  double eta_a;
  double cd;
};
} // namespace omni::hardware

#endif
