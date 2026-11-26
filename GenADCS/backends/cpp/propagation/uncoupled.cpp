#include "uncoupled.hpp"
#include "attitude_propagators/simulation_sampling.hpp"

#include "orbital_propagators/keplerian_orbit_propagator.hpp"
#include "attitude_propagators/reference_attitude_propagator.hpp"
#include "attitude_propagators/rk4_attitude_propagator.hpp"
#include "../environment/native_models.hpp"
#include "../helpers/math/time/conversions.hpp"

#include "coupled_propagators/integrate.hpp"
#include <numeric>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>
#include <stdexcept>

namespace omni::propagation {
SeparateTrajectory integrate_uncoupled(
    const Eigen::VectorXd &initial, double mu, double initial_julian_date,
    double dt, double tf, int record_every, const std::string &mode,
    const SeparateProgress &progress, const Runtime *runtime,
    SimulationSampler *sampler, int sensor_every) {
  if (sampler && sensor_every < 1)
    throw std::invalid_argument("sensor_every must be positive");
  const bool rk4 = mode == "rk4";
  if (rk4 && !runtime)
    throw std::invalid_argument("RK4 requires a prepared runtime");
  if (!std::isfinite(dt) || dt <= 0.0) {
    throw std::invalid_argument("dt must be finite and positive seconds");
  }
  if (!std::isfinite(tf) || tf < 0.0) {
    throw std::invalid_argument("tf must be finite and nonnegative seconds");
  }
  if (record_every < 1) {
    throw std::invalid_argument("record_every must be a positive integer");
  }
  if (initial.size() < 13) {
    throw std::invalid_argument("packed state must contain at least 13 values");
  }

  const ReferenceAttitudeMode reference_mode = rk4
      ? ReferenceAttitudeMode::Nadir
      : ReferenceAttitudePropagator::parse_mode(mode);
  const ReferenceAttitudePropagator reference(dt, reference_mode);
  const std::unique_ptr<omni::environment::DE421> ephemeris =
      !rk4 && reference_mode == ReferenceAttitudeMode::Sun
          ? std::make_unique<omni::environment::DE421>()
          : nullptr;
  auto apply_reference = [&](Eigen::VectorXd &state, double elapsed) {
    Eigen::Vector3d sun_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d sun_velocity = Eigen::Vector3d::Zero();
    if (ephemeris) {
      const double tdb = omni::time::utc_jd_to_tdb_jd(
          initial_julian_date + elapsed / 86400.0);
      sun_position = 1000.0 * ephemeris->position(0, tdb);
      sun_velocity = 1000.0 * ephemeris->velocity(0, tdb);
    }
    const auto attitude = reference.attitude(
        state.segment<3>(7), state.segment<3>(10), mu, sun_position,
        sun_velocity);
    state.head<4>() = attitude.quaternion;
    state.segment<3>(4) = attitude.angular_velocity;
  };

  Eigen::VectorXd state = initial;
  const double quaternion_norm = state.head<4>().norm();
  if (!std::isfinite(quaternion_norm) || quaternion_norm <= 0.0) {
    throw std::invalid_argument("attitude quaternion must have nonzero norm");
  }
  state.head<4>() /= quaternion_norm;
  if (rk4 && runtime->wheel_commands.size() != 0)
    project(state, *runtime);
  if (!rk4) apply_reference(state, 0.0);

  const auto full_steps = static_cast<std::size_t>(std::floor(tf / dt));
  double remainder = tf - static_cast<double>(full_steps) * dt;
  if (remainder <= 1e-12 * std::max(1.0, tf)) {
    remainder = 0.0;
  }
  const std::size_t total_steps = full_steps + (remainder > 0.0 ? 1 : 0);
  const auto stride = static_cast<std::size_t>(record_every);
  const std::size_t sample_count =
      1 + total_steps / stride + (total_steps % stride != 0 ? 1 : 0);
  SeparateTrajectory trajectory;
  trajectory.first.reserve(sample_count);
  trajectory.second.resize(static_cast<Eigen::Index>(sample_count), state.size());
  trajectory.first.push_back(0.0);
  trajectory.second.row(0) = state.transpose();
  if (sampler) {
    sampler->allocate(static_cast<Eigen::Index>(sample_count));
    sampler->sample(state, 0.0, 0);
    sampler->record(0);
  }

  for (std::size_t step = 0; step < total_steps; ++step) {
    const double step_dt = step == full_steps ? remainder : dt;
    if (rk4)
      state = rk4_attitude_step(state,
          initial_julian_date + static_cast<double>(step) * dt / 86400.0,
          step_dt, *runtime);
    Eigen::Vector3d next_velocity;
    const Eigen::Vector3d next_position = keplerian_position_velocity_step(
        state.segment<3>(7), state.segment<3>(10), mu, step_dt, next_velocity);
    state.segment<3>(7) = next_position;
    state.segment<3>(10) = next_velocity;
    const double elapsed = std::min(static_cast<double>(step + 1) * dt, tf);
    if (!rk4) apply_reference(state, elapsed);

    const std::size_t completed = step + 1;
    const bool sensor_tick = sampler && completed <= full_steps &&
        completed % static_cast<std::size_t>(sensor_every) == 0;
    if (sensor_tick)
      sampler->sample(state, completed * dt, completed / sensor_every);
    if (completed % stride == 0 || completed == total_steps) {
      const auto row = static_cast<Eigen::Index>(trajectory.first.size());
      trajectory.second.row(row) = state.transpose();
      if (sensor_tick) sampler->record(row);
      trajectory.first.push_back(completed == total_steps
          ? tf : std::min(static_cast<double>(completed) * dt, tf));
    }
    if (progress) {
      progress(1);
    }
  }
  return trajectory;
}

SeparateTrajectory integrate_separate_reference(
    const Eigen::VectorXd &initial, double mu, double initial_julian_date,
    double dt, double tf, int record_every, const std::string &mode,
    const SeparateProgress &progress) {
  return integrate_uncoupled(initial, mu, initial_julian_date, dt, tf,
                             record_every, mode, progress, nullptr);
}

SeparateTrajectory integrate_separate_rk4(
    const Eigen::VectorXd &initial, double mu, double initial_julian_date,
    double dt, double tf, int record_every, const Runtime &runtime,
    const SeparateProgress &progress, SimulationSampler *sampler,
    int sensor_every) {
  if (sampler && sensor_every < 1)
    throw std::invalid_argument("sensor_every must be positive");
  if (!std::isfinite(dt) || dt <= 0.0) {
    throw std::invalid_argument("dt must be finite and positive seconds");
  }
  if (!std::isfinite(tf) || tf < 0.0) {
    throw std::invalid_argument("tf must be finite and nonnegative seconds");
  }
  if (record_every < 1) {
    throw std::invalid_argument("record_every must be a positive integer");
  }
  if (initial.size() != 13 + runtime.wheel_commands.size()) {
    throw std::invalid_argument("state has incompatible wheel momentum size");
  }
  Vector state = initial;
  const double norm = state.head<4>().norm();
  if (!std::isfinite(norm) || norm <= 0.0) {
    throw std::invalid_argument("attitude quaternion must have nonzero norm");
  }
  state.head<4>() /= norm;
  if (runtime.wheel_commands.size() != 0) {
    project(state, runtime);
  }

  const auto full_steps = static_cast<std::size_t>(std::floor(tf / dt));
  double remainder = tf - static_cast<double>(full_steps) * dt;
  if (remainder <= 1e-12 * std::max(1.0, tf)) {
    remainder = 0.0;
  }
  const std::size_t total_steps = full_steps + (remainder > 0.0 ? 1 : 0);
  const auto stride = static_cast<std::size_t>(record_every);
  const std::size_t sample_count =
      1 + total_steps / stride + (total_steps % stride != 0 ? 1 : 0);
  SeparateTrajectory trajectory;
  trajectory.first.reserve(sample_count);
  trajectory.second.resize(static_cast<Eigen::Index>(sample_count), state.size());
  trajectory.first.push_back(0.0);
  trajectory.second.row(0) = state.transpose();

  if (sampler) {
    sampler->allocate(static_cast<Eigen::Index>(sample_count));
    sampler->sample(state, 0.0, 0);
    sampler->record(0);
  }
  for (std::size_t step = 0; step < total_steps; ++step) {
    const double step_dt = step == full_steps ? remainder : dt;
    // Match the Python split step: attitude uses this step's starting orbit,
    // then Keplerian propagation advances the orbit for the next step.
    state = rk4_attitude_step(
        state, initial_julian_date + static_cast<double>(step) * dt / 86400.0,
        step_dt, runtime);
    Eigen::Vector3d next_velocity;
    const Eigen::Vector3d next_position = keplerian_position_velocity_step(
        state.segment<3>(7), state.segment<3>(10), mu, step_dt, next_velocity);
    state.segment<3>(7) = next_position;
    state.segment<3>(10) = next_velocity;

    const std::size_t completed = step + 1;
    const bool sensor_tick = sampler && completed <= full_steps &&
        completed % static_cast<std::size_t>(sensor_every) == 0;
    if (sensor_tick)
      sampler->sample(state, completed * dt, completed / sensor_every);
    if (completed % stride == 0 || completed == total_steps) {
      const auto row = static_cast<Eigen::Index>(trajectory.first.size());
      trajectory.second.row(row) = state.transpose();
      if (sensor_tick) sampler->record(row);
      trajectory.first.push_back(completed == total_steps
          ? tf : std::min(static_cast<double>(completed) * dt, tf));
    }
    if (progress) {
      progress(1);
    }
  }
  return trajectory;
}
SeparateSimulation simulate_uncoupled(
    const Eigen::VectorXd &initial, double mu, double epoch, double dt, double tf,
    int record_every, Runtime &runtime, std::vector<omni::hardware::Sensor> sensors,
    const omni::hardware::estimator::AttitudeEstimator *estimator,
    int sensor_every, std::uint64_t seed, std::set<std::string> fields,
    const std::string &attitude_mode,
    bool progress, std::size_t chunk_steps) {
  if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(tf) || tf < 0 ||
      record_every < 1 || sensor_every < 1 || chunk_steps == 0)
    throw std::invalid_argument("invalid separate simulation time or sampling interval");
  SimulationSampler sampler(runtime, std::move(sensors), estimator, epoch,
                            seed, std::move(fields));
  std::size_t completed = 0;
  const auto started = std::chrono::steady_clock::now();
  SeparateProgress update;
  if (progress) update = [&](std::size_t count) {
    completed += count;
    if (completed % chunk_steps == 0 || completed * dt >= tf) {
      const double seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count();
      std::cerr << "\rRunning Keplerian + "
                << (attitude_mode == "rk4" ? "RK4" : "reference") << ": " << completed
                << "/" << static_cast<std::size_t>(std::ceil(tf / dt))
                << " steps (" << completed / std::max(seconds, 1e-9)
                << " step/s)" << std::flush;
    }
  };
  auto trajectory = integrate_uncoupled(initial, mu, epoch, dt, tf,
      record_every, attitude_mode, update, &runtime, &sampler, sensor_every);
  if (progress) std::cerr << '\n';
  return {std::move(trajectory.first), std::move(trajectory.second),
          std::move(sampler.output)};
}
SeparateSimulation simulate_coupled(
    const Eigen::VectorXd &initial, double epoch, double dt, double tf,
    int record_every, Runtime &runtime, std::vector<omni::hardware::Sensor> sensors,
    const omni::hardware::estimator::AttitudeEstimator *estimator,
    int sensor_every, std::uint64_t seed, std::set<std::string> fields,
    bool adaptive, double rtol, Vector atol, double max_step, double min_step,
    bool progress, std::size_t chunk_steps) {
  if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(tf) || tf < 0 ||
      record_every < 1 || sensor_every < 1 || chunk_steps == 0)
    throw std::invalid_argument("invalid coupled simulation interval");
  const int stride = std::gcd(record_every, sensor_every);
  SimulationSampler sampler(runtime, std::move(sensors), estimator, epoch,
                            seed, std::move(fields));
  sampler.allocate(static_cast<Eigen::Index>(std::ceil(tf / (dt * stride))) + 2);
  Eigen::Index row = 0;
  long long last_tick = -1;
  Observer observe = [&](double time, const Vector &state, bool recorded) {
    const double position = time / (dt * sensor_every);
    const auto tick = static_cast<long long>(std::llround(position));
    const bool sensor_tick = std::abs(position - tick) <= 1e-9;
    if (sensor_tick && tick != last_tick) {
      sampler.sample(state, time, static_cast<std::uint64_t>(tick));
      last_tick = tick;
    }
    if (recorded) {
      if (sensor_tick) sampler.record(row);
      ++row;
    }
  };
  std::size_t completed = 0, reported = 0;
  const auto started = std::chrono::steady_clock::now();
  Progress update;
  if (progress) update = [&](std::size_t count) {
    completed += count;
    if (completed - reported >= chunk_steps || completed * dt >= tf) {
      const double seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count();
      std::cerr << "\rRunning coupled " << (adaptive ? "DOP853" : "RK4")
                << ": " << completed << "/"
                << static_cast<std::size_t>(std::ceil(tf / dt))
                << " steps (" << completed / std::max(seconds, 1e-9)
                << " step/s)" << std::flush;
      reported = completed;
    }
  };
  auto trajectory = adaptive
      ? integrate_dop853(initial, epoch, dt, tf, runtime, rtol, atol,
                        max_step, min_step, update, stride, observe)
      : integrate_rk4(initial, epoch, dt, tf, runtime, update, stride, observe);
  if (progress) std::cerr << '\n';
  Eigen::MatrixXd states(trajectory.states.size(), initial.size());
  for (Eigen::Index i = 0; i < states.rows(); ++i)
    states.row(i) = trajectory.states[i].transpose();
  for (auto &[name, values] : sampler.output) values.conservativeResize(row, Eigen::NoChange);
  return {std::move(trajectory.times), std::move(states), std::move(sampler.output)};
}
} // namespace omni::propagation
