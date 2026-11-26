#pragma once

#include <stdexcept>
#include <utility>

namespace omni::solver {

namespace detail {
void validate_coupled_solver_options(double final_time, int chunk_steps);
}

template <typename Propagator> struct CoupledSolver {
  Propagator propagator;
  double tf;
  bool progress;
  int progress_chunk_steps;

  CoupledSolver(Propagator propagator_value, double final_time,
                bool show_progress = false, int chunk_steps = 500)
      : propagator(std::move(propagator_value)), tf(final_time),
        progress(show_progress), progress_chunk_steps(chunk_steps) {
    detail::validate_coupled_solver_options(tf, progress_chunk_steps);
  }

  auto dt() const -> decltype(propagator.dt()) { return propagator.dt(); }

  template <typename State, typename Satellite, typename Environment,
            typename Dynamics>
  decltype(auto) step(State &&state, Satellite &&satellite,
                      Environment &&environment, Dynamics &&dynamics) const {
    return propagator.step(std::forward<State>(state),
                           std::forward<Satellite>(satellite),
                           std::forward<Environment>(environment),
                           std::forward<Dynamics>(dynamics));
  }
};

} // namespace omni::solver
