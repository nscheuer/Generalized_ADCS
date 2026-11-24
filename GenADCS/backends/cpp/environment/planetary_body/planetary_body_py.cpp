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

void bind_planetary_body_planetary_body(py::module_& module) {
    py::class_<env::PlanetaryBody>(module, "PlanetaryBody")
        .def(py::init<std::string,double,double>(),py::arg("name"),py::arg("gravitational_parameter"),py::arg("reference_radius"))
        .def_readwrite("name",&env::PlanetaryBody::name)
        .def_readwrite("gravitational_parameter",&env::PlanetaryBody::gravitational_parameter)
        .def_readwrite("reference_radius",&env::PlanetaryBody::reference_radius)
        .def("__eq__",[](const env::PlanetaryBody& a,const env::PlanetaryBody& b){return a.name==b.name&&a.gravitational_parameter==b.gravitational_parameter&&a.reference_radius==b.reference_radius;});
}
