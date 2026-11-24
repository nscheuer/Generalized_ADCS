#include "atmosphere/atmospheric_model.hpp"
#include "environment.hpp"
#include "gravity/j_gravity_model.hpp"
#include "native_models.hpp"
#include "planetary_body/planetary_body.hpp"

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

void bind_magnetic_field_model_igrf14_igrf14(py::module_ &module)
{
    py::class_<env::IGRF14, env::MagneticFieldModel, std::shared_ptr<env::IGRF14>>(module, "IGRF14")
        .def(py::init<>())
        .def("field", &env::IGRF14::field)
        .def("field_inertial_tesla", &env::IGRF14::field_inertial_tesla);
}
