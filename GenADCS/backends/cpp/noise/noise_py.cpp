#include "noise.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
PYBIND11_MODULE(_noise, m) {
  using omni::noise::Noise;
  py::class_<Noise>(m, "Noise")
      .def(py::init<std::optional<Eigen::MatrixXd>>(),
           py::arg("covariance") = py::none())
      .def_readonly("covariance", &Noise::covariance)
      .def_readonly("cholesky_factor", &Noise::cholesky_factor)
      .def_readwrite("seed", &Noise::seed)
      .def(
          "apply",
          [](Noise &self, const Eigen::VectorXd &value, double dt,
             py::object rng) {
            if (rng.is_none())
              return self.apply(value, dt);
            if (!self.cholesky_factor)
              throw std::logic_error("configure a covariance");
            if (value.size() != self.cholesky_factor->rows())
              throw std::invalid_argument(
                  "value length must match noise covariance");
            Eigen::VectorXd z = rng.attr("standard_normal")(value.size())
                                    .cast<Eigen::VectorXd>();
            return Eigen::VectorXd(value + *self.cholesky_factor * z);
          },
          py::arg("value"), py::arg("dt"), py::arg("rng") = py::none());
}
