#pragma once

#include "../coupled_propagators/_coupled_dynamics.hpp"
#include "simulation_sampling.hpp"
#include <tuple>
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

SeparateTrajectory integrate_uncoupled(
    const Eigen::VectorXd &initial, double mu, double initial_julian_date,
    double dt, double tf, int record_every, const std::string &mode,
    const SeparateProgress &progress, const Runtime *runtime,
    SimulationSampler *sampler = nullptr, int sensor_every = 1);

SeparateTrajectory integrate_separate_rk4(
    const Eigen::VectorXd &initial, double mu, double initial_julian_date,
    double dt, double tf, int record_every, const Runtime &runtime,
    const SeparateProgress &progress = {}, SimulationSampler *sampler = nullptr,
    int sensor_every = 1);
using SeparateSimulation = std::tuple<std::vector<double>, Eigen::MatrixXd, SampleRows>;
SeparateSimulation simulate_uncoupled(
    const Eigen::VectorXd &initial, double mu, double epoch, double dt, double tf,
    int record_every, Runtime &runtime, std::vector<omni::hardware::Sensor> sensors,
    const omni::hardware::estimator::AttitudeEstimator *estimator,
    int sensor_every, std::uint64_t seed, std::set<std::string> fields,
    const std::string &attitude_mode,
    bool progress = false, std::size_t chunk_steps = 500);
SeparateSimulation simulate_coupled(
    const Eigen::VectorXd &initial, double epoch, double dt, double tf,
    int record_every, Runtime &runtime, std::vector<omni::hardware::Sensor> sensors,
    const omni::hardware::estimator::AttitudeEstimator *estimator,
    int sensor_every, std::uint64_t seed, std::set<std::string> fields,
    bool adaptive, double rtol, Vector atol, double max_step, double min_step,
    bool progress, std::size_t chunk_steps);
} // namespace omni::propagation
