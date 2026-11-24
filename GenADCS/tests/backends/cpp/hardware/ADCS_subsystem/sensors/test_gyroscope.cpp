#include "../check.hpp"
#include <pybind11/numpy.h>

#include <cmath>
#include <initializer_list>

namespace {
py::array_t<double> vector(std::initializer_list<double> values) {
    py::array_t<double> result(static_cast<py::ssize_t>(values.size()));
    auto out = result.mutable_unchecked<1>();
    py::ssize_t index = 0;
    for (double value : values) out(index++) = value;
    return result;
}

py::array_t<double> covariance(double x = 1e-8, double y = 1e-8, double z = 1e-8) {
    py::array_t<double> result({3, 3});
    auto out = result.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < 3; ++i)
        for (py::ssize_t j = 0; j < 3; ++j) out(i, j) = 0.0;
    out(0, 0) = x; out(1, 1) = y; out(2, 2) = z;
    return result;
}

py::object make_sensor(const py::object& type, double limit = 100.0,
                       py::object noise = py::none(), py::object bias = py::none(),
                       py::object orientation = py::none(), py::object misalignment = py::none()) {
    if (noise.is_none()) noise = covariance();
    if (bias.is_none()) bias = vector({0.0, 0.0, 0.0});
    if (orientation.is_none()) orientation = vector({1.0, 0.0, 0.0, 0.0});
    if (misalignment.is_none()) misalignment = vector({1.0, 0.0, 0.0, 0.0});
    return type(vector({0.0, 0.0, 0.0}), orientation, misalignment, limit, noise, bias);
}

void check_vector(py::handle actual, const double* expected, py::ssize_t length,
                  double tolerance = 1e-12) {
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(actual);
    CHECK(array && array.ndim() == 1 && array.shape(0) == length);
    auto values = array.unchecked<1>();
    for (py::ssize_t i = 0; i < length; ++i)
        CHECK(std::abs(values(i) - expected[i]) <= tolerance);
}

void test_noiseless_measurements_apply_mounting_and_misalignment_rotations(const py::object& type) {
    const double half_sqrt = std::sqrt(0.5);
    py::object orientation = vector({half_sqrt, 0.0, 0.0, half_sqrt});
    py::object misalignment = vector({half_sqrt, half_sqrt, 0.0, 0.0});
    py::object sensor = make_sensor(type, 100.0, py::none(), py::none(), orientation, misalignment);
    py::object omega = vector({0.1, -0.2, 0.3});
    const double expected[] = {0.2, -0.3, 0.1};
    check_vector(sensor.attr("noiseless_biasfree_measurement")(omega), expected, 3);
    check_vector(sensor.attr("noiseless_biased_measurement")(omega), expected, 3);
}

void test_bias_is_additive_and_bias_jacobian_is_identity(const py::object& type) {
    py::object bias = vector({0.01, -0.02, 0.03});
    py::object sensor = make_sensor(type, 100.0, py::none(), bias);
    py::object omega = vector({0.1, 0.2, -0.3});
    const double expected[] = {0.11, 0.18, -0.27};
    check_vector(sensor.attr("noiseless_biased_measurement")(omega), expected, 3);
    py::object jacobian = sensor.attr("jacobian_bias")();
    auto matrix = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(jacobian);
    CHECK(matrix && matrix.ndim() == 2 && matrix.shape(0) == 3 && matrix.shape(1) == 3);
    auto values = matrix.unchecked<2>();
    for (py::ssize_t i = 0; i < 3; ++i)
        for (py::ssize_t j = 0; j < 3; ++j) CHECK(values(i, j) == (i == j ? 1.0 : 0.0));
}

void test_measurement_saturates_each_axis(const py::object& type) {
    py::object sensor = make_sensor(type, 0.25);
    py::object omega = vector({0.5, -0.4, 0.1});
    const double expected[] = {0.25, -0.25, 0.1};
    check_vector(sensor.attr("noiseless_biasfree_measurement")(omega), expected, 3);
}

void test_noise_measurements_are_finite_and_shaped(const py::object& type,
                                                   const char* method_name) {
    py::object noise = covariance(1e-6, 4e-6, 9e-6);
    py::object bias = vector({0.01, 0.0, -0.01});
    py::object sensor = make_sensor(type, 100.0, noise, bias);
    py::object result = sensor.attr(method_name)(vector({0.1, 0.2, 0.3}));
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(result);
    CHECK(array && array.ndim() == 1 && array.shape(0) == 3);
    auto values = array.unchecked<1>();
    for (py::ssize_t i = 0; i < 3; ++i) CHECK(std::isfinite(values(i)));
}

void test_rate_jacobian_and_availability_contract(const py::object& type) {
    py::object sensor = make_sensor(type);
    py::object jacobian = sensor.attr("jacobian_angular_velocity")();
    auto values = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(jacobian).unchecked<2>();
    for (py::ssize_t i = 0; i < 3; ++i)
        for (py::ssize_t j = 0; j < 3; ++j) CHECK(values(i, j) == (i == j ? 1.0 : 0.0));
    CHECK(sensor.attr("availability")().cast<bool>());
}
}

int main(int argc, char** argv) {
    py::scoped_interpreter interpreter{};
    CHECK(argc == 2);
    py::module_ module = load_adcs_subsystem_module(argv[1]);
    py::object type = module.attr("Gyroscope");
    test_noiseless_measurements_apply_mounting_and_misalignment_rotations(type);
    test_bias_is_additive_and_bias_jacobian_is_identity(type);
    test_measurement_saturates_each_axis(type);
    test_noise_measurements_are_finite_and_shaped(type, "noisy_biasfree_measurement");
    test_noise_measurements_are_finite_and_shaped(type, "noisy_biased_measurement");
    test_rate_jacobian_and_availability_contract(type);
}
