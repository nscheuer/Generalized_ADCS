#pragma once
#include "_coupled_dynamics.hpp"
#include <functional>
#include <vector>
#include <functional>
namespace omni::propagation {
struct Trajectory {
  std::vector<double> times;
  std::vector<Vector> states;
};
using Observer = std::function<void(double, const Vector &, bool)>;
using SignalCheck = std::function<void()>;
using Progress = std::function<void(std::size_t)>;
Trajectory integrate_rk4(Vector, double, double, double, const Runtime &,
                         const Progress &, std::size_t, const Observer & = {},
                         const SignalCheck & = {});
Trajectory integrate_dop853(Vector, double, double, double, const Runtime &,
                            double, Vector, double, double, const Progress &,
                            std::size_t, const Observer & = {},
                            const SignalCheck & = {});
} // namespace omni::propagation
