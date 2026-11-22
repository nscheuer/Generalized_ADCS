#ifndef ADCS_OMNI_HARDWARE_SATELLITE_PANEL_HPP
#define ADCS_OMNI_HARDWARE_SATELLITE_PANEL_HPP

#include <pybind11/pybind11.h>

namespace omni::hardware {
struct SatellitePanel {
    SatellitePanel(pybind11::object position_value,
                   pybind11::object orientation_value,
                   pybind11::object dimensions_value,
                   double eta_s_value,
                   double eta_d_value,
                   double eta_a_value,
                   double cd_value);

    pybind11::object position;
    pybind11::object orientation;
    pybind11::object dimensions;
    double eta_s;
    double eta_d;
    double eta_a;
    double cd;
};
}

#endif
