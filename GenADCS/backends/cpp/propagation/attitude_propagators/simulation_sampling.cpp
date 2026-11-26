#include "simulation_sampling.hpp"
#include "../../hardware/ADCS_subsystem/estimator/attitude_estimator/direction_correction.hpp"
#include "../../helpers/math/time/conversions.hpp"
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace omni::propagation {
namespace est = omni::hardware::estimator;
SimulationSampler::SimulationSampler(
    Runtime &runtime, std::vector<omni::hardware::Sensor> sensors,
    const est::AttitudeEstimator *estimator, double epoch, std::uint64_t seed,
    std::set<std::string> fields)
    : runtime_(runtime), epoch_(epoch), sensors_(runtime, sensors, epoch, seed),
      estimator_(estimator), scratch_(sensors_.allocate(1)) {
  if (!estimator_)
    return;
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
    std::visit(
        [&](const auto &device) {
          using T = std::decay_t<decltype(device)>;
          est::Sensor entry;
          std::string field;
          int width = 0;
          if constexpr (std::is_same_v<T, omni::hardware::sensors::Gyroscope>) {
            entry.code = 0;
            field = "gyroscope_reading";
            width = 3;
          } else if constexpr (std::is_same_v<
                                   T, omni::hardware::sensors::Magnetometer>) {
            entry.code = 1;
            field = "magnetometer_reading";
            width = 3;
            entry.saturation = device.saturation_value;
          } else if constexpr (std::is_same_v<T, omni::hardware::sensors::
                                                     AutonomousStarTracker>) {
            entry.code = 2;
            field = "autonomous_star_tracker_reading";
            width = 4;
          } else if constexpr (std::is_same_v<
                                   T,
                                   omni::hardware::sensors::CoarseSunSensor>) {
            entry.code = 3;
            field = "coarse_sun_sensor_reading";
            width = 1;
            for (int i = 0; i < 3; ++i)
              entry.normal[i] = device.normal[i];
            entry.sensitivity = device.sensitivity;
            entry.saturation = device.saturation_value;
            entry.noise_std = device.noise_std;
          } else if constexpr (std::is_same_v<
                                   T, omni::hardware::sensors::FineSunSensor>) {
            entry.code = 4;
            field = "fine_sun_sensor_reading";
            width = 2;
            entry.fov = {device.fov_half_angles[0], device.fov_half_angles[1]};
          } else if constexpr (std::is_same_v<
                                   T,
                                   omni::hardware::sensors::FineEarthSensor>) {
            entry.code = 5;
            field = "fine_earth_sensor_reading";
            width = 2;
            entry.fov = {device.fov_half_angles[0], device.fov_half_angles[1]};
          }
          if constexpr (std::is_same_v<T, omni::hardware::sensors::GNSS> ||
                        std::is_same_v<T,
                                       omni::hardware::sensors::Accelerometer>)
            return;
          else {
            entry.offset = total;
            entry.enabled = device.enabled;
            for (int i = 0; i < 4; ++i) {
              entry.orientation[i] = device.orientation[i];
              entry.misalignment[i] = device.misalignment[i];
            }
            if constexpr (!std::is_same_v<
                              T, omni::hardware::sensors::CoarseSunSensor>) {
              for (int i = 0; i < (entry.code == 2 ? 3 : width); ++i) {
                if (entry.code != 2)
                  entry.bias[i] = device.bias[i];
                for (int j = 0; j < (entry.code == 2 ? 3 : width); ++j)
                  entry.noise_cholesky(i, j) = device.noise_cholesky[i][j];
              }
            } else {
              entry.bias[0] = device.bias;
            }
            devices_.push_back(entry);
            reading_layout_.push_back({field, offsets[field], width});
            offsets[field] += width;
            total += width;
          }
        },
        sensor);
  }
  readings_.resize(total);
}
void SimulationSampler::allocate(Eigen::Index rows) {
  for (const auto &[name, values] : scratch_)
    output[name] = Eigen::MatrixXd::Constant(
        rows, values.cols(), std::numeric_limits<double>::quiet_NaN());
}
void SimulationSampler::sample(const Vector &state, double time,
                               std::uint64_t tick) {
  for (auto &[name, values] : scratch_)
    values.setConstant(std::numeric_limits<double>::quiet_NaN());
  sensors_.sample(state, time, tick, scratch_, 0);
  if (!estimator_)
    return;
  const Eigen::Vector3d pos = state.segment<3>(7);
  const double jd = epoch_ + time / 86400.0;
  Eigen::Vector3d field = Eigen::Vector3d::Zero(),
                  sun = Eigen::Vector3d::Zero();
  if (runtime_.igrf)
    field = runtime_.igrf->field_inertial_tesla(pos, jd);
  else if (runtime_.chaos)
    field = runtime_.chaos->field_inertial_tesla(pos, jd);
  if (runtime_.ephemeris)
    sun = 1000.0 * runtime_.ephemeris->position(
                       0, omni::time::utc_jd_to_tdb_jd(jd)) -
          pos;
  for (auto &device : devices_) {
    if (device.code == 1)
      device.reference = field;
    else if (device.code == 3 || device.code == 4)
      device.reference = sun;
    else if (device.code == 5)
      device.reference = -pos;
  }
  Eigen::Index offset = 0;
  for (const auto &reading : reading_layout_) {
    readings_.segment(offset, reading.width) =
        scratch_.at(reading.field)
            .row(0)
            .segment(reading.offset, reading.width)
            .transpose();
    offset += reading.width;
  }
  if (ekf_)
    est::AttitudeEKF::update_sample(estimate_, estimator_->acceleration_psd,
                                    time - previous_time_, state, readings_,
                                    devices_, scratch_, 0);
  else
    est::AttitudeMEKF::update_sample(estimate_, estimator_->acceleration_psd,
                                     time - previous_time_, state, readings_,
                                     devices_, scratch_, 0);
  previous_time_ = time;
}
void SimulationSampler::record(Eigen::Index row) {
  for (const auto &[name, values] : scratch_)
    output.at(name).row(row) = values.row(0);
}
} // namespace omni::propagation
