#pragma once

#include <optional>
#include <stdexcept>
#include <utility>

namespace omni::solver
{

template <typename Dynamics, typename Orbital, typename Attitude, typename Coupled> struct BasicSolver
{
    Dynamics dynamics;
    std::optional<Orbital> orbital;
    std::optional<Attitude> attitude;
    std::optional<Coupled> coupled;

    BasicSolver(Dynamics dynamics_value, std::optional<Orbital> orbital_value, std::optional<Attitude> attitude_value,
                std::optional<Coupled> coupled_value)
        : dynamics(std::move(dynamics_value)), orbital(std::move(orbital_value)), attitude(std::move(attitude_value)),
          coupled(std::move(coupled_value))
    {
        const bool has_orbital = orbital.has_value();
        const bool has_attitude = attitude.has_value();
        const bool has_coupled = coupled.has_value();
        if ((!has_coupled && (!has_orbital || !has_attitude)) || (has_coupled && (has_orbital || has_attitude)))
        {
            throw std::invalid_argument("provide a coupled solver or both orbital and attitude solvers");
        }
    }
};

} // namespace omni::solver
