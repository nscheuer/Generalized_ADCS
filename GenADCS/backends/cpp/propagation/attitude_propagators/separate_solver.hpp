#pragma once

#include "../coupled_propagators/_coupled_dynamics.hpp"
#include <Eigen/Dense>
#include <utility>
#include <vector>
#include <functional>

namespace omni::propagation {
using SeparateTrajectory = std::pair<std::vector<double>, Eigen::MatrixXd>;
using SeparateProgress = std::function<void(std::size_t)>;

SeparateTrajectory integrate_separate_reference(
    const Eigen::VectorXd &initial, double mu, double initial_julian_date,
    double dt, double tf, int record_every, const std::string &mode,
    const SeparateProgress &progress = {});

SeparateTrajectory integrate_separate_rk4(
    const Eigen::VectorXd &initial, double mu, double initial_julian_date,
    double dt, double tf, int record_every, const Runtime &runtime,
    const SeparateProgress &progress = {});
} // namespace omni::propagation
