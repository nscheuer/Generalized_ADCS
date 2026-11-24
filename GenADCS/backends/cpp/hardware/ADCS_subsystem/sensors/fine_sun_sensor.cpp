#include "fine_sun_sensor.hpp"
#include "kernels.hpp"
#include "philox.hpp"

#include <cmath>
#include <stdexcept>

namespace omni::hardware::sensors {
FineSunSensor::FineSunSensor(V p, Q q, Q m, M2 covariance, V2 b, V2 fov)
    : position(p), orientation(kernels::normalize(q)), misalignment(kernels::normalize(m)),
      noise(covariance), bias(b), fov_half_angles(fov) {
    const double a = noise[0][0];
    const double off_diagonal = 0.5 * (noise[0][1] + noise[1][0]);
    const double d = noise[1][1];
    const double remainder = d - off_diagonal * off_diagonal / a;
    if (!(a > 0.0) || !(d > 0.0) ||
        std::abs(noise[0][1] - noise[1][0]) > 1e-12 * (1.0 + std::abs(off_diagonal)) ||
        !(remainder > 0.0))
        throw std::invalid_argument("noise covariance must be positive definite");
    noise_cholesky = {{{{std::sqrt(a), 0.0}},
                       {{off_diagonal / std::sqrt(a), std::sqrt(remainder)}}}};
    const double half_pi = std::acos(-1.0) / 2.0;
    for (double angle : fov_half_angles)
        if (!(angle > 0.0) || angle > half_pi)
            throw std::invalid_argument("FOV half-angles must be positive and <= pi/2");
}

FineSunSensor::V2 FineSunSensor::measure(const V& sun, const Q& attitude, bool with_bias, bool with_noise) {
    V2 draw{};
    if (with_noise) {
        const auto values = omni::helpers::random::standard_normals(seed, random_tick++, 0x40000000U, 2);
        draw = {{values[0], values[1]}};
    }
    return kernels::fine_direction_measurement(sun, attitude, orientation, misalignment, bias,
                                                noise_cholesky, with_bias, with_noise,
                                                with_noise ? &draw : nullptr);
}

FineSunSensor::V2 FineSunSensor::noiseless_biasfree_measurement(const V& s, const Q& q) const {
    return const_cast<FineSunSensor*>(this)->measure(s, q, false, false);
}
FineSunSensor::V2 FineSunSensor::noiseless_biased_measurement(const V& s, const Q& q) const {
    return const_cast<FineSunSensor*>(this)->measure(s, q, true, false);
}
FineSunSensor::V2 FineSunSensor::noisy_biasfree_measurement(const V& s, const Q& q) { return measure(s, q, false, true); }
FineSunSensor::V2 FineSunSensor::noisy_biased_measurement(const V& s, const Q& q) { return measure(s, q, true, true); }
FineSunSensor::M2 FineSunSensor::jacobian_bias() const { return {{{{1, 0}}, {{0, 1}}}}; }
FineSunSensor::M23 FineSunSensor::jacobian_relative_quaternion(const V& s, const Q& q, const Q& d) const {
    return kernels::fine_sun_attitude_jacobian(s, q, orientation, misalignment, d, 0);
}
FineSunSensor::M23 FineSunSensor::jacobian_relative_mrp(const V& s, const Q& q, const V& d) const {
    return kernels::fine_sun_attitude_jacobian(s, q, orientation, misalignment, d, 1);
}
FineSunSensor::M23 FineSunSensor::jacobian_relative_angle_vector(const V& s, const Q& q, const V& d) const {
    return kernels::fine_sun_attitude_jacobian(s, q, orientation, misalignment, d, 2);
}
bool FineSunSensor::availability(const V& s, const Q& q, bool visible) const {
    return kernels::fine_direction_available(s, q, orientation, misalignment, fov_half_angles, enabled, visible);
}
FineSunSensor::M3 FineSunSensor::fisher_information_matrix(const V& s, const Q& q, bool visible) const {
    if (!availability(s, q, visible)) return {};
    const V sensor_sun = kernels::rotate_inertial_to_sensor(s, q, orientation, misalignment);
    return kernels::fine_sun_fisher_information(
        {{std::atan2(sensor_sun[0], sensor_sun[2]), std::atan2(sensor_sun[1], sensor_sun[2])}},
        noise, orientation, misalignment);
}
}
