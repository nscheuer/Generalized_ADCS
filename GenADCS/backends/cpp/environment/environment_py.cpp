#include "environment.hpp"
#include "gravity/j_gravity_model.hpp"
#include "planetary_body/planetary_body.hpp"
#include "atmosphere/atmospheric_model.hpp"
#include "native_models.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <array>
#include <string>
#include <utility>

namespace py = pybind11;
namespace env = omni::environment;

void bind_environment(py::module_& module) {
    py::class_<env::Environment>(module,"Environment").def(py::init([](py::object body,py::object atmosphere,py::object magnetic,py::object gravity,py::object ephemeris){if(body.is_none())body=py::cast(env::Earth());return env::Environment(std::move(body),std::move(atmosphere),std::move(magnetic),std::move(gravity),std::move(ephemeris));}),py::arg("planetary_body")=py::none(),py::arg("atmosphere")=py::none(),py::arg("magnetic_field")=py::none(),py::arg("j_gravity")=py::none(),py::arg("ephemeris")=py::none()).def_readwrite("planetary_body",&env::Environment::planetary_body).def_readwrite("atmosphere",&env::Environment::atmosphere).def_readwrite("magnetic_field",&env::Environment::magnetic_field).def_readwrite("j_gravity",&env::Environment::j_gravity).def_readwrite("ephemeris",&env::Environment::ephemeris);
}
