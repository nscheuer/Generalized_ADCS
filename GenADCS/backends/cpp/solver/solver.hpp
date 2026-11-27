#pragma once

#include <optional>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace omni::solver {

template <typename Dynamics, typename Orbital, typename Attitude,
          typename Coupled>
struct BasicSolver {
  Dynamics dynamics;
  std::optional<Orbital> orbital;
  std::optional<Attitude> attitude;
  std::optional<Coupled> coupled;
  std::optional<double> tf;
  bool progress;
  int progress_chunk_steps;

  BasicSolver(Dynamics dynamics_value, std::optional<Orbital> orbital_value,
              std::optional<Attitude> attitude_value,
              std::optional<Coupled> coupled_value,
              std::optional<double> final_time = std::nullopt,
              bool show_progress = true, int chunk_steps = 500)
      : dynamics(std::move(dynamics_value)), orbital(std::move(orbital_value)),
        attitude(std::move(attitude_value)), coupled(std::move(coupled_value)),
        tf(final_time), progress(show_progress),
        progress_chunk_steps(chunk_steps) {
    const bool has_orbital = orbital.has_value();
    const bool has_attitude = attitude.has_value();
    const bool has_coupled = coupled.has_value();
    if ((!has_coupled && (!has_orbital || !has_attitude)) ||
        (has_coupled && (has_orbital || has_attitude))) {
      throw std::invalid_argument(
          "provide a coupled solver or both orbital and attitude solvers");
    }
    if (!has_coupled &&
        (!tf.has_value() || !std::isfinite(*tf) || *tf < 0.0)) {
      throw std::invalid_argument(
          "separate solvers require a finite nonnegative tf");
    }
    if (progress_chunk_steps <= 0) {
      throw std::invalid_argument(
          "progress_chunk_steps must be a positive integer");
    }
  }
};

} // namespace omni::solver
