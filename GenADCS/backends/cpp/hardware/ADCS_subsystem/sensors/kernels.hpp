#ifndef GENADCS_HARDWARE_SENSORS_KERNELS_HPP
#define GENADCS_HARDWARE_SENSORS_KERNELS_HPP
#include "../../../helpers/math/rotations/sensor_kernels.hpp"
namespace omni::hardware::sensors::kernels {
V rate_measurement(const V &angular_velocity_body, const Q &orientation,
                   const Q &misalignment, const V &bias,
                   const M &noise_cholesky, double saturation, bool with_bias,
                   bool with_noise, const V *noise_draw);
V vector_measurement(const V &vector_inertial, const Q &attitude,
                     const Q &orientation, const Q &misalignment, const V &bias,
                     const M &noise_cholesky, double saturation, bool with_bias,
                     bool with_noise, const V *noise_draw);
M magnetometer_information(const V &field, const Q &attitude,
                           const Q &orientation, const Q &misalignment,
                           const M &noise, const M &field_covariance);
V accelerometer_measurement(const V &acceleration_inertial,
                            const V &gravity_inertial, const Q &attitude,
                            const Q &orientation, const Q &misalignment,
                            const V &bias, const M &noise_cholesky,
                            double saturation, bool with_bias, bool with_noise,
                            const V *noise_draw);
double scalar_sun_measurement(const V &sun, const Q &attitude,
                              const Q &orientation, const Q &misalignment,
                              const V &normal, double sensitivity, double bias,
                              double noise_std, double saturation,
                              bool with_bias, bool with_noise,
                              double noise_draw);
bool coarse_sun_available(const V &sun, const Q &attitude, const Q &orientation,
                          const Q &misalignment, const V &normal, double fov,
                          bool enabled, bool visible);
V coarse_sun_attitude_jacobian(const V &sun, const Q &attitude,
                               const Q &orientation, const Q &misalignment,
                               const V &normal, double sensitivity,
                               const Q &delta, int kind);
V coarse_sun_attitude_jacobian(const V &sun, const Q &attitude,
                               const Q &orientation, const Q &misalignment,
                               const V &normal, double sensitivity,
                               const V &delta, int kind);
M coarse_sun_information(const V &sun, const Q &attitude, const Q &orientation,
                         const Q &misalignment, const V &normal,
                         double sensitivity, double variance);
std::array<double, 2> fine_direction_measurement(
    const V &sun, const Q &attitude, const Q &orientation,
    const Q &misalignment, const std::array<double, 2> &bias,
    const std::array<std::array<double, 2>, 2> &noise_cholesky, bool with_bias,
    bool with_noise, const std::array<double, 2> *draw);
bool fine_direction_available(const V &sun, const Q &attitude,
                              const Q &orientation, const Q &misalignment,
                              const std::array<double, 2> &fov, bool enabled,
                              bool visible);
std::array<std::array<double, 3>, 2>
fine_sun_attitude_jacobian(const V &sun, const Q &attitude,
                           const Q &orientation, const Q &misalignment,
                           const Q &delta, int kind);
std::array<std::array<double, 3>, 2>
fine_sun_attitude_jacobian(const V &sun, const Q &attitude,
                           const Q &orientation, const Q &misalignment,
                           const V &delta, int kind);
M fine_sun_fisher_information(
    const std::array<double, 2> &angles,
    const std::array<std::array<double, 2>, 2> &covariance,
    const Q &orientation, const Q &misalignment);
} // namespace omni::hardware::sensors::kernels
#endif
