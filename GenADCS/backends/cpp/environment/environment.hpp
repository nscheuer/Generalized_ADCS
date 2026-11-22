#ifndef ADCS_OMNI_ENVIRONMENT_ENVIRONMENT_HPP
#define ADCS_OMNI_ENVIRONMENT_ENVIRONMENT_HPP
#include <pybind11/pybind11.h>
namespace omni::environment {
struct Environment {
    Environment(pybind11::object planetary_body, pybind11::object atmosphere,
                pybind11::object magnetic_field, pybind11::object j_gravity,
                pybind11::object ephemeris);
    pybind11::object planetary_body, atmosphere, magnetic_field, j_gravity, ephemeris;
};
}
#endif
