#include "orbital_solver.hpp"
#include "../propagation/orbital_propagators/keplerian_orbit_propagator.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using NativeKeplerianSolver = omni::solver::OrbitalSolver<omni::propagation::KeplerianOrbitPropagator>;
void bind_orbital_solver(py::module_ &module) {
  py::class_<NativeKeplerianSolver>(module, "OrbitalSolver")
      .def(py::init<omni::propagation::KeplerianOrbitPropagator>(), py::arg("propagator"))
      .def_readwrite("propagator", &NativeKeplerianSolver::propagator)
      .def_property_readonly("dt", &NativeKeplerianSolver::dt)
      .def("step", [](const NativeKeplerianSolver &self, const Eigen::VectorXd &state, double mu) {
        return self.step(state, mu);
      }, py::arg("state"), py::arg("mu"))
      .def("integrate", [](const NativeKeplerianSolver &self, const Eigen::VectorXd &state, double mu, double tf, int record_every) {
        return self.integrate(state, mu, tf, record_every);
      }, py::arg("state"), py::arg("mu"), py::arg("tf"), py::arg("record_every") = 1);
}
