#ifndef GENADCS_HARDWARE_SENSORS_FINE_EARTH_SENSOR_HPP
#define GENADCS_HARDWARE_SENSORS_FINE_EARTH_SENSOR_HPP

#include "fine_sun_sensor.hpp"

namespace omni::hardware::sensors
{
// Earth and Sun use the same two-axis direction measurement model.
struct FineEarthSensor : FineSunSensor
{
    using FineSunSensor::FineSunSensor;
};
} // namespace omni::hardware::sensors

#endif
