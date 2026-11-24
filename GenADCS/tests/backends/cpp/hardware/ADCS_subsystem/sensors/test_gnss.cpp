#include "gnss.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

using omni::hardware::sensors::GNSS;
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #condition); } while (false)

int main() {
    GNSS::M6 noise{}; for (int i=0;i<6;++i) noise[i][i]=static_cast<double>(i+1);
    GNSS::V6 bias{{1,2,3,.1,.2,.3}};
    GNSS sensor(noise,bias);
    GNSS::V p{{1,2,3}}, v{{-4,5,6}};
    auto clean=sensor.noiseless_biasfree_measurement(p,v);
    GNSS::V6 expected{{1,2,3,-4,5,6}};
    CHECK(clean==expected);
    auto biased=sensor.noiseless_biased_measurement(p,v);
    for(int i=0;i<6;++i) CHECK(std::abs(biased[i]-expected[i]-bias[i])<1e-12);
    auto noisy_biasfree=sensor.noisy_biasfree_measurement(p,v);
    auto noisy_biased=sensor.noisy_biased_measurement(p,v);
    for(double x:noisy_biasfree) CHECK(std::isfinite(x));
    for(double x:noisy_biased) CHECK(std::isfinite(x));
    auto jp=sensor.jacobian_position(), jv=sensor.jacobian_velocity();
    for(int i=0;i<6;++i) for(int j=0;j<3;++j) {
        CHECK(jp[i][j]==(i<3 && i==j ? 1.0 : 0.0));
        CHECK(jv[i][j]==(i>=3 && i-3==j ? 1.0 : 0.0));
    }
    for(auto jac : {sensor.jacobian_relative_quaternion(), sensor.jacobian_relative_mrp(), sensor.jacobian_relative_angle_vector()})
        for(const auto& row:jac) for(double x:row) CHECK(x==0.0);
    const auto jb=sensor.jacobian_bias();
    for(int i=0;i<6;++i) for(int j=0;j<6;++j) CHECK(jb[i][j]==(i==j ? 1.0 : 0.0));
    CHECK(sensor.availability()); sensor.enabled=false; CHECK(!sensor.availability());
}
