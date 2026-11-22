#include "environment.hpp"
#include <utility>
namespace omni::environment {
Environment::Environment(pybind11::object body, pybind11::object atmosphere_model,
                         pybind11::object magnetic_model, pybind11::object gravity_model,
                         pybind11::object ephemeris_model)
    : planetary_body(std::move(body)), atmosphere(std::move(atmosphere_model)),
      magnetic_field(std::move(magnetic_model)), j_gravity(std::move(gravity_model)),
      ephemeris(std::move(ephemeris_model)) {}
}
