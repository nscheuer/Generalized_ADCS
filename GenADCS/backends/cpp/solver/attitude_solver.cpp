#include "attitude_solver.hpp"

#include "../propagation/attitude_propagators/rk4_attitude_propagator.hpp"

#include <cmath>
#include <stdexcept>

namespace omni::solver {

AttitudeSolver::AttitudeSolver(double step_seconds)
    : propagator_(RK4AttitudeConfiguration{step_seconds}) {
  if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
    throw std::invalid_argument("dt must be finite and positive seconds");
  }
}

AttitudeSolver::AttitudeSolver(double step_seconds, const std::string &mode)
    : propagator_(propagation::ReferenceAttitudePropagator(
          step_seconds,
          propagation::ReferenceAttitudePropagator::parse_mode(mode))) {}

double AttitudeSolver::dt() const noexcept {
  if (const auto *rk4 = std::get_if<RK4AttitudeConfiguration>(&propagator_)) {
    return rk4->dt;
  }
  return std::get<propagation::ReferenceAttitudePropagator>(propagator_).dt();
}

bool AttitudeSolver::is_reference() const noexcept {
  return std::holds_alternative<propagation::ReferenceAttitudePropagator>(
      propagator_);
}

int AttitudeSolver::mode() const noexcept {
  if (!is_reference()) {
    return -1;
  }
  return static_cast<int>(
      std::get<propagation::ReferenceAttitudePropagator>(propagator_).mode());
}

propagation::Trajectory AttitudeSolver::integrate_rk4(
    const propagation::Vector &state, double jd, double tf,
    const propagation::Runtime &runtime, std::size_t record_every) const {
  const auto *rk4 = std::get_if<RK4AttitudeConfiguration>(&propagator_);
  if (!rk4) {
    throw std::logic_error("integrate_rk4 requires an RK4 attitude solver");
  }
  return propagation::integrate_attitude_rk4(state, jd, rk4->dt, tf, runtime,
                                             record_every);
}

void AttitudeSolver::apply_reference(
    Eigen::Ref<Eigen::MatrixXd> states, const std::vector<double> &times,
    double jd, double mu) const {
  const auto *reference =
      std::get_if<propagation::ReferenceAttitudePropagator>(&propagator_);
  if (!reference) {
    throw std::logic_error("apply_reference requires a reference solver");
  }
  reference->apply(states, times, jd, mu);
}

} // namespace omni::solver
