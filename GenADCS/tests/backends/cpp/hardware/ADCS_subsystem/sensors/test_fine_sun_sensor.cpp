#include "fine_sun_sensor.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #x); } while(false)

using Sensor = omni::hardware::sensors::FineSunSensor;
using V = Sensor::V;
using Q = Sensor::Q;
using V2 = Sensor::V2;
using M2 = Sensor::M2;

void close(double actual, double expected, double tolerance = 1e-8) {
    CHECK(std::abs(actual - expected) <= tolerance);
}
Q axis_angle(V axis, double angle) {
    const double length = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    const double scale = std::sin(angle/2)/length;
    return {{std::cos(angle/2), axis[0]*scale, axis[1]*scale, axis[2]*scale}};
}
Sensor make_sensor(Q orientation = {{1,0,0,0}}, Q misalignment = {{1,0,0,0}},
                   M2 covariance = M2{{{{.01,0}},{{0,.02}}}},
                   V2 bias = V2{{.01,-.02}}, V2 fov = V2{{.2,.3}}) {
    return Sensor({{0,0,0}}, orientation, misalignment, covariance, bias, fov);
}

int main() {
    const Q identity{{1,0,0,0}};
    const V sun{{.1,.2,.97}};
    Sensor sensor = make_sensor();
    const auto mounted = sensor.noiseless_biasfree_measurement(sun, identity);
    close(mounted[0], std::atan2(sun[0], sun[2]));
    close(mounted[1], std::atan2(sun[1], sun[2]));
    const auto biased = sensor.noiseless_biased_measurement(sun, identity);
    close(biased[0], mounted[0] + sensor.bias[0]);
    close(biased[1], mounted[1] + sensor.bias[1]);
    const auto bias_jacobian = sensor.jacobian_bias();
    CHECK(bias_jacobian[0][0] == 1 && bias_jacobian[0][1] == 0);
    CHECK(bias_jacobian[1][0] == 0 && bias_jacobian[1][1] == 1);

    Sensor aligned = make_sensor();
    const auto jq = aligned.jacobian_relative_quaternion(sun, identity, identity);
    const V zero{{0,0,0}};
    const auto jm = aligned.jacobian_relative_mrp(sun, identity, zero);
    const auto ja = aligned.jacobian_relative_angle_vector(sun, identity, zero);
    for (int i=0;i<2;++i) for (int j=0;j<3;++j) {
        close(jm[i][j], 2*jq[i][j]);
        close(ja[i][j], .5*jq[i][j]);
    }
    CHECK(aligned.availability({{0,0,1}}, identity));
    CHECK(!aligned.availability({{0,0,1}}, identity, false));
    CHECK(!aligned.availability({{std::sin(.25),0,std::cos(.25)}}, identity));
    aligned.enabled = false;
    CHECK(!aligned.availability({{0,0,1}}, identity));
    aligned.enabled = true;
    const auto info = aligned.fisher_information_matrix({{0,0,1}}, identity);
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        close(info[i][j], info[j][i]);
        CHECK(std::isfinite(info[i][j]));
    }
    const auto unavailable_info = aligned.fisher_information_matrix({{0,0,-1}}, identity);
    for (const auto& row : unavailable_info) for (double value : row) close(value, 0);
    aligned.seed = 42;
    const auto noisy = aligned.noisy_biasfree_measurement(sun, identity);
    CHECK(aligned.random_tick == 1);
    Sensor repeated = make_sensor(); repeated.seed = 42;
    const auto repeated_noisy = repeated.noisy_biasfree_measurement(sun, identity);
    close(noisy[0], repeated_noisy[0]); close(noisy[1], repeated_noisy[1]);

    bool invalid_covariance = false;
    try { make_sensor(identity, identity, M2{{{{1,2}},{{2,1}}}}); }
    catch (const std::invalid_argument&) { invalid_covariance = true; }
    CHECK(invalid_covariance);
    bool invalid_fov = false;
    try { make_sensor(identity, identity, M2{{{{1,0}},{{0,1}}}}, V2{{0,0}}, V2{{0,.3}}); }
    catch (const std::invalid_argument&) { invalid_fov = true; }
    CHECK(invalid_fov);
}
