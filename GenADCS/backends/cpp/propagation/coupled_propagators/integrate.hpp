#pragma once
#include "_coupled_dynamics.hpp"
#include <functional>
#include <vector>
namespace omni::propagation
{
struct Trajectory
{
    std::vector<double> times;
    std::vector<Vector> states;
};
using Progress = std::function<void(std::size_t)>;
Trajectory integrate_rk4(Vector, double, double, double, const Runtime &, const Progress &, std::size_t);
Trajectory integrate_dop853(Vector, double, double, double, const Runtime &, double, Vector, double, double,
                            const Progress &, std::size_t);
} // namespace omni::propagation
