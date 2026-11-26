#include "rk4_attitude_propagator.hpp"
#include "../coupled_propagators/runtime_python.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <utility>

namespace py = pybind11;
using namespace omni::propagation;

void bind_rk4_attitude_propagator(py::module_ &m) {
  m.def("rk4_attitude_step",
        [](Vector state, double jd, double dt, py::object runtime) {
          return rk4_attitude_step(state, jd, dt, runtime_from_python(runtime));
        });
  m.def("integrate_attitude_rk4",
        [](Vector state, double jd, double dt, double tf, py::object runtime,
           std::size_t record_every) {
          const auto result = integrate_attitude_rk4(
              std::move(state), jd, dt, tf, runtime_from_python(runtime),
              record_every);
          py::array_t<double> times(result.times.size());
          py::array_t<double> states(
              {static_cast<py::ssize_t>(result.states.size()),
               static_cast<py::ssize_t>(result.states.front().size())});
          auto t = times.mutable_unchecked<1>();
          auto s = states.mutable_unchecked<2>();
          for (std::size_t i = 0; i < result.states.size(); ++i) {
            t(i) = result.times[i];
            for (Eigen::Index j = 0; j < result.states[i].size(); ++j) {
              s(i, j) = result.states[i][j];
            }
          }
          return py::make_tuple(times, states);
        },
        py::arg("state"), py::arg("jd"), py::arg("dt"), py::arg("tf"),
        py::arg("runtime"), py::arg("record_every") = 1);
}
