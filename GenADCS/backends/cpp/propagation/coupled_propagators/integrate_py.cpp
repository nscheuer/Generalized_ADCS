#include "integrate.hpp"
#include "runtime_python.hpp"
#include "sensor_sampling.hpp"
#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace omni::propagation;
namespace {
py::tuple output(const Trajectory &result) {
  py::array_t<double> times(result.times.size());
  py::array_t<double> states(
      {static_cast<py::ssize_t>(result.states.size()),
       static_cast<py::ssize_t>(result.states.front().size())});
  auto t = times.mutable_unchecked<1>();
  auto s = states.mutable_unchecked<2>();
  for (std::size_t i = 0; i < result.times.size(); ++i) {
    t(i) = result.times[i];
    for (Eigen::Index j = 0; j < result.states[i].size(); ++j)
      s(i, j) = result.states[i][j];
  }
  return py::make_tuple(times, states);
}
struct ProgressBar {
  py::object bar = py::none();
  ProgressBar(bool enabled, double total, const char *description) {
    if (enabled)
      bar = py::module_::import("tqdm").attr("tqdm")(
          py::arg("total") = total, py::arg("desc") = description,
          py::arg("unit") = "step");
  }
  Progress callback() {
    if (bar.is_none())
      return {};
    return [this](std::size_t n) { bar.attr("update")(n); };
  }
  ~ProgressBar() {
    if (!bar.is_none()) {
      try {
        bar.attr("close")();
      } catch (py::error_already_set &e) {
        e.discard_as_unraisable("progress close");
      }
    }
  }
};
} // namespace
void bind_integrate(py::module_ &m) {
  m.def("sample_sensors",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> times,
           const Eigen::MatrixXd &states, double jd, double interval,
           py::object runtime, py::object adcs, std::uint64_t seed) {
          auto r = runtime_from_python(runtime);
          auto input = times.unchecked<1>();
          Eigen::VectorXd native_times(input.shape(0));
          for (py::ssize_t i = 0; i < input.shape(0); ++i)
            native_times[i] = input(i);
          std::vector<omni::hardware::Sensor> sensors;
          for (py::handle item : adcs.attr("sensors")) {
            namespace hw = omni::hardware;
            namespace sen = omni::hardware::sensors;
            if (py::isinstance<sen::Gyroscope>(item))
              sensors.emplace_back(item.cast<sen::Gyroscope>());
            else if (py::isinstance<sen::Magnetometer>(item))
              sensors.emplace_back(item.cast<sen::Magnetometer>());
            else if (py::isinstance<sen::Accelerometer>(item))
              sensors.emplace_back(item.cast<sen::Accelerometer>());
            else if (py::isinstance<sen::CoarseSunSensor>(item))
              sensors.emplace_back(item.cast<sen::CoarseSunSensor>());
            else if (py::isinstance<sen::FineEarthSensor>(item))
              sensors.emplace_back(item.cast<sen::FineEarthSensor>());
            else if (py::isinstance<sen::FineSunSensor>(item))
              sensors.emplace_back(item.cast<sen::FineSunSensor>());
            else if (py::isinstance<sen::GNSS>(item))
              sensors.emplace_back(item.cast<sen::GNSS>());
            else if (py::isinstance<sen::AutonomousStarTracker>(item))
              sensors.emplace_back(item.cast<sen::AutonomousStarTracker>());
            else
              throw py::type_error("unsupported C++ sensor");
          }
          return sample_sensors(native_times, states, jd, interval, r,
                                std::move(sensors), seed);
        });
  m.def(
      "integrate_rk4",
      [](Vector y, double jd, double dt, double tf, py::object p, bool progress,
         std::size_t every) {
        auto r = runtime_from_python(p);
        ProgressBar bar(progress, dt > 0 ? std::ceil(tf / dt) : 0,
                        "Running coupled RK4");
        return output(integrate_rk4(y, jd, dt, tf, r, bar.callback(), every));
      },
      py::arg("y"), py::arg("jd"), py::arg("dt"), py::arg("tf"),
      py::arg("runtime"), py::arg("progress") = false,
      py::arg("record_every") = 1);
  m.def(
      "integrate_dop853",
      [](Vector y, double jd, double dt, double tf, py::object p, double rtol,
         Vector atol, double max_step, double min_step, bool progress,
         std::size_t every) {
        auto r = runtime_from_python(p);
        ProgressBar bar(progress, dt > 0 ? std::ceil(tf / dt) : 0,
                        "Running coupled DOP853");
        return output(integrate_dop853(y, jd, dt, tf, r, rtol, atol, max_step,
                                       min_step, bar.callback(), every));
      },
      py::arg("y"), py::arg("jd"), py::arg("dt"), py::arg("tf"),
      py::arg("runtime"), py::arg("rtol"), py::arg("atol"), py::arg("max_step"),
      py::arg("min_step"), py::arg("progress") = false,
      py::arg("record_every") = 0);
}
