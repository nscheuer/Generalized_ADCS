#include "simulation_sampling.hpp"
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace omni::propagation {
namespace est = omni::hardware::estimator;
SimulationSampler::SimulationSampler(Runtime &runtime,
    std::vector<omni::hardware::Sensor> sensors,
    const est::AttitudeEstimator *estimator, double epoch, std::uint64_t seed,
    std::set<std::string> fields)
    : sensors_(runtime, sensors, epoch, seed), estimator_(estimator),
      scratch_(sensors_.allocate(1)) {
  if (!estimator_) return;
  ekf_ = dynamic_cast<const est::AttitudeEKF *>(estimator_) != nullptr;
  if (!ekf_ && !dynamic_cast<const est::AttitudeMEKF *>(estimator_))
    throw std::invalid_argument("unsupported native attitude estimator");
  estimate_ = estimator_->prepare();
  fields.insert(estimator_->record.begin(), estimator_->record.end());
  auto diagnostics = ekf_ ? est::AttitudeEKF::allocate_samples(1, fields)
                          : est::AttitudeMEKF::allocate_samples(1, fields);
  scratch_.insert(diagnostics.begin(), diagnostics.end());
  std::map<std::string, int> offsets;
  int total = 0;
  for (const auto &sensor : sensors) {
    std::visit([&](const auto &s) {
      using T = std::decay_t<decltype(s)>;
      if constexpr (std::is_same_v<T, omni::hardware::sensors::Gyroscope> ||
                    std::is_same_v<T, omni::hardware::sensors::AutonomousStarTracker>) {
        constexpr bool gyro = std::is_same_v<T, omni::hardware::sensors::Gyroscope>;
        const std::string field = gyro ? "gyroscope_reading" : "autonomous_star_tracker_reading";
        const int width = gyro ? 3 : 4;
        est::Sensor device;
        device.code = gyro ? 0 : 2;
        device.offset = total;
        device.enabled = s.enabled;
        for (int i = 0; i < 4; ++i) {
          device.orientation[i] = s.orientation[i];
          device.misalignment[i] = s.misalignment[i];
        }
        for (int i = 0; i < 3; ++i) {
          device.bias[i] = s.bias[i];
          for (int j = 0; j < 3; ++j)
            device.noise_cholesky(i, j) = s.noise_cholesky[i][j];
        }
        devices_.push_back(device);
        reading_layout_.push_back({field, offsets[field], width});
        offsets[field] += width;
        total += width;
      }
    }, sensor);
  }
  readings_.resize(total);
}
void SimulationSampler::allocate(Eigen::Index rows) {
  for (const auto &[name, values] : scratch_)
    output[name] = Eigen::MatrixXd::Constant(rows, values.cols(),
                                           std::numeric_limits<double>::quiet_NaN());
}
void SimulationSampler::sample(const Vector &state, double time, std::uint64_t tick) {
  for (auto &[name, values] : scratch_)
    values.setConstant(std::numeric_limits<double>::quiet_NaN());
  sensors_.sample(state, time, tick, scratch_, 0);
  if (!estimator_) return;
  Eigen::Index offset = 0;
  for (const auto &reading : reading_layout_) {
    readings_.segment(offset, reading.width) =
        scratch_.at(reading.field).row(0).segment(reading.offset, reading.width).transpose();
    offset += reading.width;
  }
  if (ekf_)
    est::AttitudeEKF::update_sample(estimate_, estimator_->acceleration_psd,
        time - previous_time_, state, readings_, devices_, scratch_, 0);
  else
    est::AttitudeMEKF::update_sample(estimate_, estimator_->acceleration_psd,
        time - previous_time_, state, readings_, devices_, scratch_, 0);
  previous_time_ = time;
}
void SimulationSampler::record(Eigen::Index row) {
  for (const auto &[name, values] : scratch_)
    output.at(name).row(row) = values.row(0);
}
} // namespace omni::propagation
