#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_HPP

#include "magnetorquer.hpp"
#include "reaction_wheel.hpp"

#include "sensors/accelerometer.hpp"
#include "sensors/autonomous_star_tracker.hpp"
#include "sensors/coarse_sun_sensor.hpp"
#include "sensors/fine_earth_sensor.hpp"
#include "sensors/fine_sun_sensor.hpp"
#include "sensors/gnss.hpp"
#include "sensors/gyroscope.hpp"
#include "sensors/magnetometer.hpp"
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

namespace omni::hardware {
namespace estimator {
class AttitudeEstimator;
}
using Sensor = std::variant<sensors::Gyroscope, sensors::Magnetometer,
                            sensors::Accelerometer, sensors::CoarseSunSensor,
                            sensors::FineSunSensor, sensors::FineEarthSensor,
                            sensors::GNSS, sensors::AutonomousStarTracker>;
struct ADCSSubsystem {
  ADCSSubsystem() = default;
  ADCSSubsystem(
      std::vector<ReactionWheel> reaction_wheels_value,
      std::vector<Magnetorquer> magnetorquers_value,
      std::vector<Sensor> sensors_value = {}, int N_value = 1,
      std::shared_ptr<estimator::AttitudeEstimator> estimator_value = nullptr);

  bool every_step(std::uint64_t tick) const;
  std::vector<Sensor> sensors;
  int N = 1;
  std::shared_ptr<estimator::AttitudeEstimator> estimator;
  std::vector<ReactionWheel> reaction_wheels;
  std::vector<Magnetorquer> magnetorquers;
};
} // namespace omni::hardware

#endif
