#include "hardware/ADCS_subsystem/estimator/attitude_estimator/base.hpp"
#include "../../check.hpp"
#include <stdexcept>
using namespace omni::hardware::estimator;
int main() {
    Eigen::Vector4d q(2,0,0,0);
    AttitudeEstimator e(q, Eigen::Vector3d::Zero(), Matrix6::Identity(), Eigen::Matrix3d::Identity(), {"kalman_gain"});
    CHECK((e.quaternion-Eigen::Vector4d(1,0,0,0)).norm()<1e-14);
    auto state=e.prepare(); state.rate.x()=7;
    CHECK(e.prepare().rate.x()==0);
    bool rejected=false;
    try { AttitudeEstimator bad(Eigen::Vector4d::Zero(),Eigen::Vector3d::Zero(),Matrix6::Identity(),Eigen::Matrix3d::Identity()); }
    catch (const std::invalid_argument&) { rejected=true; }
    CHECK(rejected);
    rejected=false;
    try { AttitudeEstimator bad(q,Eigen::Vector3d::Zero(),-Matrix6::Identity(),Eigen::Matrix3d::Identity()); }
    catch (const std::invalid_argument&) { rejected=true; }
    CHECK(rejected);
    rejected=false;
    try { AttitudeEstimator bad(q,Eigen::Vector3d::Zero(),Matrix6::Identity(),Eigen::Matrix3d::Identity(),{"invalid"}); }
    catch (const std::invalid_argument&) { rejected=true; }
    CHECK(rejected);
}
