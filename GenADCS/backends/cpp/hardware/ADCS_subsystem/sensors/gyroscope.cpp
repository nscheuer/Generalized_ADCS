#include "gyroscope.hpp"
#include "kernels.hpp"
#include "philox.hpp"

#include <stdexcept>

namespace omni::hardware::sensors {
Gyroscope::Gyroscope(V p, Q q, Q mis, double saturation, M covariance, V b)
    : position(p), orientation(kernels::normalize(q)),
      misalignment(kernels::normalize(mis)), saturation_value(saturation),
      noise(covariance), noise_cholesky(kernels::cholesky(noise)), bias(b) {
    if (!(saturation_value > 0.0))
        throw std::invalid_argument("saturation_value must be positive");
}

Gyroscope::V Gyroscope::measure(const V& rate, bool with_bias, bool with_noise) {
    V draw{};
    if (with_noise) {
        const auto values = omni::helpers::random::standard_normals(seed, random_tick++, 0x40000000U, 3);
        for (std::size_t i = 0; i < draw.size(); ++i) draw[i] = values[i];
    }
    return kernels::rate_measurement(rate, orientation, misalignment, bias,
        noise_cholesky, saturation_value, with_bias, with_noise,
        with_noise ? &draw : nullptr);
}

Gyroscope::V Gyroscope::noiseless_biasfree_measurement(const V& rate) const {
    return const_cast<Gyroscope*>(this)->measure(rate, false, false);
}
Gyroscope::V Gyroscope::noiseless_biased_measurement(const V& rate) const {
    return const_cast<Gyroscope*>(this)->measure(rate, true, false);
}
Gyroscope::V Gyroscope::noisy_biasfree_measurement(const V& rate) { return measure(rate, false, true); }
Gyroscope::V Gyroscope::noisy_biased_measurement(const V& rate) { return measure(rate, true, true); }

Gyroscope::M Gyroscope::jacobian_angular_velocity() const {
    return {{{{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}}};
}
Gyroscope::M Gyroscope::jacobian_bias() const { return jacobian_angular_velocity(); }
bool Gyroscope::availability() const { return enabled; }
}
