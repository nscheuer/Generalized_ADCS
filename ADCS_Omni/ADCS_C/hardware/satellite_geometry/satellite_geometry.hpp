#ifndef ADCS_OMNI_HARDWARE_SATELLITE_GEOMETRY_HPP
#define ADCS_OMNI_HARDWARE_SATELLITE_GEOMETRY_HPP

#include <pybind11/pybind11.h>

namespace omni::hardware {
struct SatelliteGeometry {
    SatelliteGeometry(double mass_value,
                      pybind11::object inertia_value,
                      pybind11::list panels_value,
                      pybind11::object center_of_mass_value);

    double mass;
    pybind11::object inertia;
    pybind11::list panels;
    pybind11::object center_of_mass;
};
}

#endif
