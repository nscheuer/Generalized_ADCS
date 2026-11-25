#include "bias.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>

namespace py = pybind11;
using omni::bias::Bias;
using omni::bias::Matrix;
using omni::bias::Vector;

namespace {
Vector to_vector(py::handle value) {
  auto array =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          value);
  if (!array) {
    throw py::type_error("value must be convertible to a numeric array");
  }
  auto info = array.request();
  Vector result(static_cast<std::size_t>(info.size));
  const auto *data = static_cast<const double *>(info.ptr);
  std::copy(data, data + info.size, result.begin());
  return result;
}
py::object to_python(const Vector &value) {
  if (value.size() == 1) {
    return py::float_(value[0]);
  }
  py::array_t<double> result(value.size());
  std::copy(value.begin(), value.end(), result.mutable_data());
  return result;
}
py::object matrix_object(const Matrix &matrix, const char *form,
                         const char *coordinates) {
  py::array_t<double> array({matrix.size(), matrix.size()});
  auto output = array.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < output.shape(0); ++i) {
    for (py::ssize_t j = 0; j < output.shape(1); ++j) {
      output(i, j) =
          matrix[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    }
  }
  auto covariance = py::module_::import("ADCS.covariance").attr("Covariance");
  return covariance(array, py::arg("form") = form,
                    py::arg("coordinates") = coordinates);
}
} // namespace

PYBIND11_MODULE(_bias, module) {
  module.doc() = "Native C++ bias model bindings";
  py::class_<Bias>(module, "Bias")
      .def(
          py::init([](py::object bias, py::object std_bias, py::object bounds) {
            auto pair = py::cast<py::tuple>(bounds);
            return Bias(to_vector(bias), to_vector(std_bias),
                        to_vector(pair[0]), to_vector(pair[1]));
          }),
          py::arg("bias") = 0.0, py::arg("std_bias") = 0.0,
          py::arg("bounds") =
              py::make_tuple(-std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity()))
      .def_property(
          "bias", [](const Bias &self) { return to_python(self.bias()); },
          [](Bias &self, py::object value) { self.set_bias(to_vector(value)); })
      .def_property(
          "std_bias",
          [](const Bias &self) { return to_python(self.std_bias()); },
          [](Bias &self, py::object value) {
            self.set_std_bias(to_vector(value));
          })
      .def_property(
          "estimated_std_bias",
          [](const Bias &self) { return to_python(self.estimated_std_bias()); },
          [](Bias &self, py::object value) {
            self.set_estimated_std_bias(to_vector(value));
          })
      .def_property(
          "bounds",
          [](const Bias &self) {
            return py::make_tuple(to_python(self.lower_bounds()),
                                  to_python(self.upper_bounds()));
          },
          [](Bias &self, py::tuple bounds) {
            self.set_bounds(to_vector(bounds[0]), to_vector(bounds[1]));
          })
      .def_property("last_bias_time", &Bias::last_bias_time,
                    &Bias::set_last_bias_time)
      .def("__bool__", [](const Bias &self) { return static_cast<bool>(self); })
      .def("copy", &Bias::copy)
      .def("broadcast_to", &Bias::broadcast_to, py::arg("length"),
           py::kw_only(), py::arg("owner") = "sensor")
      .def(
          "process_covariance",
          [](const Bias &self, double dt, const std::string &form) {
            return matrix_object(self.process_covariance(dt), form.c_str(),
                                 "bias");
          },
          py::arg("dt") = 1.0, py::kw_only(), py::arg("form") = "full")
      .def(
          "process_psd",
          [](const Bias &self, const std::string &form) {
            return matrix_object(self.process_psd(), form.c_str(), "bias_rate");
          },
          py::kw_only(), py::arg("form") = "full")
      .def(
          "_update_bias",
          [](Bias &self, double j2000, py::object rng) {
            const double previous = self.last_bias_time();
            if (!std::isfinite(previous)) {
              self.set_last_bias_time(j2000);
              return;
            }
            const double elapsed =
                std::max(0.0, (j2000 - previous) * 3155760000.0);
            Vector draw(self.bias().size());
            if (!rng.is_none()) {
              draw = to_vector(rng.attr("normal")(0.0, 1.0, draw.size()));
            } else {
              static thread_local std::mt19937_64 engine(
                  std::random_device{}());
              std::normal_distribution<double> normal;
              for (auto &x : draw) {
                x = normal(engine);
              }
            }
            self.update(j2000, elapsed, draw);
          },
          py::arg("j2000"), py::arg("rng") = py::none())
      .def(
          "get_bias",
          [](const Bias &self, double) { return to_python(self.get_bias()); },
          py::arg("j2000"))
      .def(
          "apply",
          [](const Bias &self, py::object value, double, py::object) {
            return to_python(self.apply(to_vector(value)));
          },
          py::arg("value"), py::arg("dt"), py::arg("rng") = py::none());
}
