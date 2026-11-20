#include "satellite_panel.hpp"

#include <utility>

namespace omni::hardware {
SatellitePanel::SatellitePanel(pybind11::object position_value,
                               pybind11::object orientation_value,
                               pybind11::object dimensions_value,
                               double eta_s_value,
                               double eta_d_value,
                               double eta_a_value,
                               double cd_value)
    : position(std::move(position_value)),
      orientation(std::move(orientation_value)),
      dimensions(std::move(dimensions_value)),
      eta_s(eta_s_value),
      eta_d(eta_d_value),
      eta_a(eta_a_value),
      cd(cd_value) {}
}
