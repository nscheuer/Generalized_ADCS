#include "environment.hpp"

#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace env = omni::environment;

void bind_environment(py::module_ &module)
{
    py::class_<env::Environment>(module, "Environment")
        .def(py::init<std::shared_ptr<env::PlanetaryBody>, std::shared_ptr<env::AtmosphericModel>,
                      std::shared_ptr<env::MagneticFieldModel>, std::shared_ptr<env::JGravityModel>,
                      std::shared_ptr<env::DE421>>(),
             py::arg("planetary_body") = nullptr, py::arg("atmosphere") = nullptr, py::arg("magnetic_field") = nullptr,
             py::arg("j_gravity") = nullptr, py::arg("ephemeris") = nullptr)
        .def_readwrite("planetary_body", &env::Environment::planetary_body)
        .def_readwrite("atmosphere", &env::Environment::atmosphere)
        .def_readwrite("magnetic_field", &env::Environment::magnetic_field)
        .def_readwrite("j_gravity", &env::Environment::j_gravity)
        .def_readwrite("ephemeris", &env::Environment::ephemeris);
}
