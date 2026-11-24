#include "noise/noise.hpp"
#include "../hardware/ADCS_subsystem/check.hpp"
#include <stdexcept>
int main() {
    omni::noise::Noise noise(Eigen::MatrixXd::Identity(3,3));
    CHECK(noise.cholesky_factor->isIdentity());
    CHECK(noise.sample(12,3).isApprox(noise.sample(12,3)));
    CHECK(!noise.sample(12,3).isApprox(noise.sample(12,4)));
    CHECK(noise.apply(Eigen::Vector3d::Ones(),1).isApprox(Eigen::Vector3d::Ones()+noise.sample(0,0)));
    bool rejected=false;
    try { omni::noise::Noise bad(-Eigen::MatrixXd::Identity(3,3)); }
    catch (const std::invalid_argument&) { rejected=true; }
    CHECK(rejected);
}
