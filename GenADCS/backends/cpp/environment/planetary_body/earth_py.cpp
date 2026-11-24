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

void bind_planetary_body_earth(py::module_& module) {
    py::class_<env::Earth,env::PlanetaryBody,std::shared_ptr<env::Earth>>(module,"Earth").def(py::init<>());
}
