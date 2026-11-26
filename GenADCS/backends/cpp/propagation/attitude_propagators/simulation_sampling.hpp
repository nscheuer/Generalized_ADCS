#pragma once
#include "../../hardware/ADCS_subsystem/estimator/attitude_estimator/attitude_ekf.hpp"
#include "../../hardware/ADCS_subsystem/estimator/attitude_estimator/attitude_sequential_ukf.hpp"
#include "../coupled_propagators/sensor_sampling.hpp"

namespace omni::propagation {
// Owned by one simulation. No Python objects or callbacks enter this runtime.
class SimulationSampler {
public:
  SimulationSampler(
      Runtime &runtime, std::vector<omni::hardware::Sensor> sensors,
      const omni::hardware::estimator::AttitudeEstimator *estimator,
      double epoch, std::uint64_t seed, std::set<std::string> fields);
  void allocate(Eigen::Index rows);
  void sample(const Vector &state, double time, std::uint64_t tick);
  void record(Eigen::Index row);
  SampleRows output;

private:
  Runtime &runtime_;
  double epoch_;
  SensorSampler sensors_;
  const omni::hardware::estimator::AttitudeEstimator *estimator_;
  bool ekf_ = false;
  const omni::hardware::estimator::AttitudeSequentialUKF *ukf_ = nullptr;
  omni::hardware::estimator::State estimate_;
  double previous_time_ = 0.0;
  std::vector<omni::hardware::estimator::Sensor> devices_;
  struct Reading {
    std::string field;
    int offset;
    int width;
  };
  std::vector<Reading> reading_layout_;
  Eigen::VectorXd readings_;
  SampleRows scratch_;
};
} // namespace omni::propagation
