#include "separate_solver.hpp"

#include "../orbital_propagators/keplerian_orbit_propagator.hpp"
#include "reference_attitude_propagator.hpp"
#include "rk4_attitude_propagator.hpp"

#include <cmath>
#include <stdexcept>

namespace omni::propagation {
SeparateTrajectory integrate_separate_reference(
    const Eigen::VectorXd &initial, double mu, double initial_julian_date,
    double dt, double tf, int record_every, const std::string &mode,
    const SeparateProgress &progress) {
  auto trajectory = KeplerianOrbitPropagator(dt).integrate(
      initial, mu, tf, record_every, progress);
  ReferenceAttitudePropagator(dt,
      ReferenceAttitudePropagator::parse_mode(mode))
      .apply(trajectory.second, trajectory.first, initial_julian_date, mu);
  return trajectory;
}

SeparateTrajectory integrate_separate_rk4(
    const Eigen::VectorXd &initial, double mu, double initial_julian_date,
    double dt, double tf, int record_every, const Runtime &runtime,
    const SeparateProgress &progress) {
  auto trajectory = KeplerianOrbitPropagator(dt).integrate(
      initial, mu, tf, record_every, progress);
  const auto attitude = integrate_attitude_rk4(initial, initial_julian_date,
      dt, tf, runtime, static_cast<std::size_t>(record_every));
  if (trajectory.first.size() != attitude.times.size()) {
    throw std::runtime_error("separate propagators returned different sample counts");
  }
  for (std::size_t row = 0; row < attitude.times.size(); ++row) {
    if (std::abs(trajectory.first[row] - attitude.times[row]) > 1e-9) {
      throw std::runtime_error("separate propagators returned mismatched sample times");
    }
    trajectory.second.row(static_cast<Eigen::Index>(row)).segment<7>(0) =
        attitude.states[row].segment<7>(0).transpose();
    if (initial.size() > 13) {
      trajectory.second.row(static_cast<Eigen::Index>(row)).rightCols(
          initial.size() - 13) = attitude.states[row].tail(initial.size() - 13).transpose();
    }
  }
  return trajectory;
}
} // namespace omni::propagation
