#include "gyroscope.hpp"
#include "philox.hpp"
#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;
namespace omni::hardware::sensors {

Gyroscope::Gyroscope(py::handle position_value, py::handle orientation_value,
                     py::handle misalignment_value, double saturation,
                     py::handle noise_value, py::handle bias_value)
    : position(kernels::read_vector3(position_value, "position")),
      orientation(kernels::normalize(kernels::read_quaternion(orientation_value, "orientation"))),
      misalignment(kernels::normalize(kernels::read_quaternion(misalignment_value, "misalignment"))),
      saturation_value(saturation), noise(kernels::read_matrix(noise_value, "noise")),
      noise_cholesky(kernels::cholesky(noise)), bias(kernels::read_vector3(bias_value, "bias")) {
    if (!(saturation_value > 0.0)) throw py::value_error("saturation_value must be positive");
}

py::array_t<double> Gyroscope::measure(py::handle rate, bool with_bias, bool with_noise) {
    Vector3 draw{};
    if (with_noise) {
        const auto values = omni::helpers::random::standard_normals(seed, random_tick++, 0x40000000U, 3);
        for (std::size_t i = 0; i < 3; ++i) draw[i] = values[i];
    }
    return kernels::array_vector(kernels::rate_measurement(
        kernels::read_vector3(rate, "angular_velocity_body"), orientation, misalignment,
        bias, noise_cholesky, saturation_value, with_bias, with_noise,
        with_noise ? &draw : nullptr));
}
py::array_t<double> Gyroscope::noiseless_biasfree_measurement(py::handle rate) const { return const_cast<Gyroscope*>(this)->measure(rate, false, false); }
py::array_t<double> Gyroscope::noiseless_biased_measurement(py::handle rate) const { return const_cast<Gyroscope*>(this)->measure(rate, true, false); }
py::array_t<double> Gyroscope::noisy_biasfree_measurement(py::handle rate) { return measure(rate, false, true); }
py::array_t<double> Gyroscope::noisy_biased_measurement(py::handle rate) { return measure(rate, true, true); }
py::array_t<double> Gyroscope::jacobian_angular_velocity() const {
    py::array_t<double> result({3, 3}); auto out = result.mutable_unchecked<2>();
    for (py::ssize_t i=0; i<3; ++i) for (py::ssize_t j=0; j<3; ++j) out(i,j) = (i == j ? 1.0 : 0.0);
    return result;
}
py::array_t<double> Gyroscope::jacobian_bias() const { return jacobian_angular_velocity(); }
bool Gyroscope::availability() const { return enabled; }
}
