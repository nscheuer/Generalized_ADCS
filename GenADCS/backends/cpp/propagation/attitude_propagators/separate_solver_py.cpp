#include "separate_solver.hpp"
#include "../coupled_propagators/runtime_python.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>

namespace py = pybind11;

void bind_separate_solver(py::module_ &module) {
  module.def("integrate_separate_reference",
             [](const Eigen::VectorXd &initial, double mu, double epoch,
                double dt, double tf, int record_every,
                const std::string &mode, bool progress,
                py::object progress_callback, std::size_t chunk_steps) {
               omni::propagation::SeparateProgress callback;
               std::size_t pending = 0;
               if (chunk_steps == 0) throw py::value_error("progress_chunk_steps must be positive");
               callback = [&pending, chunk_steps, progress, progress_callback](std::size_t completed) {
                 pending += completed;
                 if (pending >= chunk_steps) {
                   if (PyErr_CheckSignals() != 0) throw py::error_already_set();
                   if (progress) progress_callback(pending);
                   pending = 0;
                 }
               };
               auto result = omni::propagation::integrate_separate_reference(
                   initial, mu, epoch, dt, tf, record_every, mode, callback);
               if (progress && pending) progress_callback(pending);
               return result;
             },
             py::arg("initial"), py::arg("mu"), py::arg("initial_julian_date"),
             py::arg("dt"), py::arg("tf"), py::arg("record_every"),
             py::arg("mode"), py::arg("progress") = false,
             py::arg("progress_callback") = py::none(),
             py::arg("progress_chunk_steps") = 500);
  module.def("integrate_separate_rk4",
             [](const Eigen::VectorXd &initial, double mu,
                double initial_julian_date, double dt, double tf,
                int record_every, py::object runtime, bool progress,
                py::object progress_callback, std::size_t chunk_steps) {
               omni::propagation::SeparateProgress callback;
               std::size_t pending = 0;
               if (chunk_steps == 0) throw py::value_error("progress_chunk_steps must be positive");
               callback = [&pending, chunk_steps, progress, progress_callback](std::size_t completed) {
                 pending += completed;
                 if (pending >= chunk_steps) {
                   if (PyErr_CheckSignals() != 0) throw py::error_already_set();
                   if (progress) progress_callback(pending);
                   pending = 0;
                 }
               };
               auto result = omni::propagation::integrate_separate_rk4(
                   initial, mu, initial_julian_date, dt, tf, record_every,
                   omni::propagation::runtime_from_python(runtime), callback);
               if (progress && pending) progress_callback(pending);
               return result;
             }, py::arg("initial"), py::arg("mu"),
             py::arg("initial_julian_date"), py::arg("dt"), py::arg("tf"),
             py::arg("record_every"), py::arg("runtime"),
             py::arg("progress") = false,
             py::arg("progress_callback") = py::none(),
             py::arg("progress_chunk_steps") = 500);
}
