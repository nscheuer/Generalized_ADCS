#include "fine_sun_sensor.hpp"
#include "kernels.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #x); } while(false)

using Sensor = omni::hardware::sensors::FineSunSensor;
using V = Sensor::V;
using Q = Sensor::Q;
using V2 = Sensor::V2;
using M2 = Sensor::M2;
namespace kernels = omni::hardware::sensors::kernels;

void close(double actual, double expected, double tolerance = 1e-8) {
    CHECK(std::abs(actual - expected) <= tolerance);
}
Q axis_angle(V axis, double angle) {
    const double length = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    const double scale = std::sin(angle/2)/length;
    return {{std::cos(angle/2), axis[0]*scale, axis[1]*scale, axis[2]*scale}};
}
Sensor make_sensor(Q orientation = {{1,0,0,0}}, Q misalignment = {{1,0,0,0}}) {
    return Sensor({{0,0,0}}, orientation, misalignment,
                  M2{{{{.01,0}},{{0,.02}}}}, V2{{.01,-.02}}, V2{{.2,.3}});
}
int main() {
    const Q identity{{1,0,0,0}};
    const V sun{{.1,.2,.97}};
    Sensor sensor = make_sensor(axis_angle({{0,1,0}},.2), axis_angle({{1,0,0}},-.1));
    const auto mounted = sensor.native_measurement(sun, identity, false, false);
    const V rotated = kernels::multiply(kernels::quaternion_matrix(
        kernels::multiply(sensor.misalignment, sensor.orientation)), sun);
    close(mounted[0], std::atan2(rotated[0], rotated[2]));
    close(mounted[1], std::atan2(rotated[1], rotated[2]));
    const auto biased = sensor.native_measurement(sun, identity, true, false);
    close(biased[0], mounted[0] + sensor.bias[0]);
    close(biased[1], mounted[1] + sensor.bias[1]);
    const auto bias_jacobian = sensor.native_jacobian_bias();
    CHECK(bias_jacobian[0][0] == 1 && bias_jacobian[0][1] == 0);
    CHECK(bias_jacobian[1][0] == 0 && bias_jacobian[1][1] == 1);

    Sensor aligned = make_sensor();
    const auto jq = aligned.native_jacobian_quaternion(sun, identity, identity);
    const V zero{{0,0,0}};
    const auto jm = aligned.native_jacobian_mrp(sun, identity, zero);
    const auto ja = aligned.native_jacobian_angle(sun, identity, zero);
    for (int i=0;i<2;++i) for (int j=0;j<3;++j) {
        close(jm[i][j], 2*jq[i][j]);
        close(ja[i][j], .5*jq[i][j]);
    }
    CHECK(aligned.native_availability({{0,0,1}}, identity));
    CHECK(!aligned.native_availability({{0,0,1}}, identity, false));
    CHECK(!aligned.native_availability({{std::sin(.25),0,std::cos(.25)}}, identity));
    aligned.enabled = false;
    CHECK(!aligned.native_availability({{0,0,1}}, identity));
    aligned.enabled = true;
    const auto info = aligned.native_fisher_information({{0,0,1}}, identity);
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) close(info[i][j], info[j][i]);
    const auto unavailable_info = aligned.native_fisher_information({{0,0,-1}}, identity);
    for (const auto& row : unavailable_info) for (double value : row) close(value, 0);
    aligned.seed = 42;
    const auto noisy = aligned.native_measurement(sun, identity, false, true);
    CHECK(aligned.random_tick == 1);
    Sensor repeated = make_sensor();
    repeated.seed = 42;
    const auto repeated_noisy = repeated.native_measurement(sun, identity, false, true);
    close(noisy[0], repeated_noisy[0]);
    close(noisy[1], repeated_noisy[1]);
}
