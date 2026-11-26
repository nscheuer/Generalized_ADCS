#include "separate_solver.hpp"
#include "simulation_sampling.hpp"
#include "../coupled_propagators/runtime_python.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace py = pybind11;

namespace {
namespace hardware = omni::hardware;
hardware::Sensor as_sensor(py::handle value) {
  namespace sensors = hardware::sensors;
  // Check the derived sensor before its base to preserve its concrete type.
  if (py::isinstance<sensors::FineEarthSensor>(value)) {
    return hardware::Sensor(std::in_place_type<sensors::FineEarthSensor>,
                            value.cast<const sensors::FineEarthSensor &>());
  }
  if (py::isinstance<sensors::FineSunSensor>(value)) {
    return hardware::Sensor(std::in_place_type<sensors::FineSunSensor>,
                            value.cast<const sensors::FineSunSensor &>());
  }
  if (py::isinstance<sensors::Gyroscope>(value)) {
    return hardware::Sensor(std::in_place_type<sensors::Gyroscope>,
                            value.cast<const sensors::Gyroscope &>());
  }
  if (py::isinstance<sensors::Magnetometer>(value)) {
    return hardware::Sensor(std::in_place_type<sensors::Magnetometer>,
                            value.cast<const sensors::Magnetometer &>());
  }
  if (py::isinstance<sensors::Accelerometer>(value)) {
    return hardware::Sensor(std::in_place_type<sensors::Accelerometer>,
                            value.cast<const sensors::Accelerometer &>());
  }
  if (py::isinstance<sensors::CoarseSunSensor>(value)) {
    return hardware::Sensor(std::in_place_type<sensors::CoarseSunSensor>,
                            value.cast<const sensors::CoarseSunSensor &>());
  }
  if (py::isinstance<sensors::GNSS>(value)) {
    return hardware::Sensor(std::in_place_type<sensors::GNSS>,
                            value.cast<const sensors::GNSS &>());
  }
  if (py::isinstance<sensors::AutonomousStarTracker>(value)) {
    return hardware::Sensor(
        std::in_place_type<sensors::AutonomousStarTracker>,
        value.cast<const sensors::AutonomousStarTracker &>());
  }
  throw py::type_error("sensors must contain native C++ sensor objects");
}

} // namespace

void bind_separate_solver(py::module_ &module) {
  module.def("simulate_separate_rk4",
      [](const Eigen::VectorXd &initial, double mu, double epoch,
         double dt, double tf, int record_every, py::object runtime,
         py::object adcs, std::uint64_t seed, std::set<std::string> fields,
         bool progress, std::size_t chunk_steps) {
        if (chunk_steps == 0)
          throw py::value_error("progress_chunk_steps must be positive");
        auto native_runtime = omni::propagation::runtime_from_python(runtime);
        std::vector<hardware::Sensor> sensors;
        for (py::handle sensor : adcs.attr("sensors"))
          sensors.push_back(as_sensor(sensor));
        const py::object estimator_object = adcs.attr("estimator");
        const auto *estimator = estimator_object.is_none() ? nullptr :
            estimator_object.cast<const hardware::estimator::AttitudeEstimator *>();
        const int sensor_every = adcs.attr("N").cast<int>();
        py::gil_scoped_release release;
        return omni::propagation::simulate_separate_rk4(initial, mu, epoch,
            dt, tf, record_every, native_runtime, std::move(sensors), estimator,
            sensor_every, seed, std::move(fields), progress, chunk_steps);
      }, py::arg("initial"), py::arg("mu"), py::arg("initial_julian_date"),
      py::arg("dt"), py::arg("tf"), py::arg("record_every"), py::arg("runtime"),
      py::arg("adcs"), py::arg("seed"), py::arg("fields"),
      py::arg("progress") = false, py::arg("progress_chunk_steps") = 500);
  module.def("simulate_coupled",
      [](const Eigen::VectorXd &initial, double epoch,
         double dt, double tf, int record_every, py::object runtime,
         py::object adcs, std::uint64_t seed, std::set<std::string> fields,
         bool adaptive, double rtol, Eigen::VectorXd atol, double max_step, double min_step,
         bool progress, std::size_t chunk_steps) {
        if (chunk_steps == 0)
          throw py::value_error("progress_chunk_steps must be positive");
        auto native_runtime = omni::propagation::runtime_from_python(runtime);
        std::vector<hardware::Sensor> sensors;
        for (py::handle sensor : adcs.attr("sensors"))
          sensors.push_back(as_sensor(sensor));
        const py::object estimator_object = adcs.attr("estimator");
        const auto *estimator = estimator_object.is_none() ? nullptr :
            estimator_object.cast<const hardware::estimator::AttitudeEstimator *>();
        const int sensor_every = adcs.attr("N").cast<int>();
        py::gil_scoped_release release;
        return omni::propagation::simulate_coupled(initial, epoch,
            dt, tf, record_every, native_runtime, std::move(sensors), estimator,
            sensor_every, seed, std::move(fields), adaptive, rtol, atol, max_step, min_step, progress, chunk_steps);
      }, py::arg("initial"), py::arg("initial_julian_date"),
      py::arg("dt"), py::arg("tf"), py::arg("record_every"), py::arg("runtime"),
      py::arg("adcs"), py::arg("seed"), py::arg("fields"),
      py::arg("adaptive"), py::arg("rtol"), py::arg("atol"),
      py::arg("max_step"), py::arg("min_step"),
      py::arg("progress") = false, py::arg("progress_chunk_steps") = 500);
  module.def("integrate_separate_reference",
             [](const Eigen::VectorXd &initial, double mu, double epoch,
                double dt, double tf, int record_every,
                const std::string &mode, bool progress,
                std::size_t chunk_steps) {
               if (chunk_steps == 0) throw py::value_error("progress_chunk_steps must be positive");
               std::size_t completed = 0;
               const auto started = std::chrono::steady_clock::now();
               omni::propagation::SeparateProgress update;
               if (progress) update = [&](std::size_t count) {
                 completed += count;
                 if (completed % chunk_steps == 0 || completed * dt >= tf) {
                   const double seconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - started).count();
                   std::cerr << "\rRunning Keplerian + reference: " << completed
                             << "/" << static_cast<std::size_t>(std::ceil(tf / dt))
                             << " steps (" << completed / std::max(seconds, 1e-9)
                             << " step/s)" << std::flush;
                 }
               };
               py::gil_scoped_release release;
               auto result = omni::propagation::integrate_separate_reference(
                   initial, mu, epoch, dt, tf, record_every, mode, update);
               if (progress) std::cerr << '\n';
               return result;
             },
             py::arg("initial"), py::arg("mu"), py::arg("initial_julian_date"),
             py::arg("dt"), py::arg("tf"), py::arg("record_every"),
             py::arg("mode"), py::arg("progress") = false,
             py::arg("progress_chunk_steps") = 500);
  module.def("integrate_separate_rk4",
             [](const Eigen::VectorXd &initial, double mu,
                double initial_julian_date, double dt, double tf,
                int record_every, py::object runtime, bool progress,
                py::object progress_callback, std::size_t chunk_steps) {
               if (chunk_steps == 0) throw py::value_error("progress_chunk_steps must be positive");
               auto result = omni::propagation::integrate_separate_rk4(
                   initial, mu, initial_julian_date, dt, tf, record_every,
                   omni::propagation::runtime_from_python(runtime));
               // Notify Python only after the native integration has completed.
               if (progress && !progress_callback.is_none())
                 progress_callback(static_cast<std::size_t>(std::ceil(tf / dt)));
               return result;
             }, py::arg("initial"), py::arg("mu"),
             py::arg("initial_julian_date"), py::arg("dt"), py::arg("tf"),
             py::arg("record_every"), py::arg("runtime"),
             py::arg("progress") = false,
             py::arg("progress_callback") = py::none(),
             py::arg("progress_chunk_steps") = 500);
}
