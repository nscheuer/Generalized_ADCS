#include "magnetometer.hpp"
#include "kernels.hpp"
#include "philox.hpp"

#include <stdexcept>

namespace omni::hardware::sensors
{
Magnetometer::Magnetometer(V p, Q q, Q mis, double saturation, M covariance, V b)
    : position(p), orientation(kernels::normalize(q)), misalignment(kernels::normalize(mis)),
      saturation_value(saturation), noise(covariance), noise_cholesky(kernels::cholesky(covariance)), bias(b)
{
    if (!(saturation_value > 0.0))
        throw std::invalid_argument("saturation_value must be positive");
}

Magnetometer::V Magnetometer::sensor_field(const V &field, const Q &attitude) const
{
    return kernels::rotate_inertial_to_sensor(field, attitude, orientation, misalignment);
}

Magnetometer::V Magnetometer::measure(const V &field, const Q &attitude, bool with_bias, bool with_noise)
{
    V draw{};
    if (with_noise)
    {
        const auto values = omni::helpers::random::standard_normals(seed, random_tick++, 0x40000000U, 3);
        for (std::size_t i = 0; i < 3; ++i)
            draw[i] = values[i];
    }
    return kernels::vector_measurement(field, attitude, orientation, misalignment, bias, noise_cholesky,
                                       saturation_value, with_bias, with_noise, with_noise ? &draw : nullptr);
}

Magnetometer::V Magnetometer::noiseless_biasfree_measurement(const V &f, const Q &q) const
{
    return const_cast<Magnetometer *>(this)->measure(f, q, false, false);
}
Magnetometer::V Magnetometer::noiseless_biased_measurement(const V &f, const Q &q) const
{
    return const_cast<Magnetometer *>(this)->measure(f, q, true, false);
}
Magnetometer::V Magnetometer::noisy_biasfree_measurement(const V &f, const Q &q)
{
    return measure(f, q, false, true);
}
Magnetometer::V Magnetometer::noisy_biased_measurement(const V &f, const Q &q)
{
    return measure(f, q, true, true);
}
Magnetometer::M Magnetometer::jacobian_bias() const
{
    return kernels::identity();
}

Magnetometer::M Magnetometer::jacobian_relative_quaternion(const V &field, const Q &attitude, const Q &delta) const
{
    return kernels::attitude_vector_jacobian_quaternion(sensor_field(field, attitude), delta);
}
Magnetometer::M Magnetometer::jacobian_relative_mrp(const V &field, const Q &attitude, const V &delta) const
{
    return kernels::attitude_vector_jacobian_mrp(sensor_field(field, attitude), delta);
}
Magnetometer::M Magnetometer::jacobian_relative_angle_vector(const V &field, const Q &attitude, const V &delta) const
{
    return kernels::attitude_vector_jacobian_angle(sensor_field(field, attitude), delta);
}
bool Magnetometer::availability() const
{
    return enabled;
}
Magnetometer::M Magnetometer::fisher_information_matrix(const V &field, const Q &attitude,
                                                        const M &field_covariance) const
{
    if (!enabled)
        return {};
    return kernels::magnetometer_information(field, attitude, orientation, misalignment, noise, field_covariance);
}
} // namespace omni::hardware::sensors
