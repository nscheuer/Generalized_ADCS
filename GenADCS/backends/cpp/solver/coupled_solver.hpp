#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>

namespace omni::solver
{

template <typename Propagator> struct CoupledSolver
{
    Propagator propagator;
    double tf;
    bool progress;
    int progress_chunk_steps;

    CoupledSolver(Propagator propagator_value, double final_time, bool show_progress = false, int chunk_steps = 500)
        : propagator(std::move(propagator_value)), tf(final_time), progress(show_progress),
          progress_chunk_steps(chunk_steps)
    {
        if (!std::isfinite(tf) || tf < 0.0)
        {
            throw std::invalid_argument("tf must be finite and nonnegative seconds");
        }
        if (progress_chunk_steps <= 0)
        {
            throw std::invalid_argument("progress_chunk_steps must be a positive integer");
        }
    }

    auto dt() const -> decltype(propagator.dt())
    {
        return propagator.dt();
    }

    template <typename State, typename Satellite, typename Environment, typename Dynamics>
    decltype(auto) step(State &&state, Satellite &&satellite, Environment &&environment, Dynamics &&dynamics) const
    {
        return propagator.step(std::forward<State>(state), std::forward<Satellite>(satellite),
                               std::forward<Environment>(environment), std::forward<Dynamics>(dynamics));
    }
};

} // namespace omni::solver
