#include "hardware/ADCS_subsystem/estimator/attitude_estimator/attitude_mekf.hpp"
#include "../../check.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
using namespace omni::hardware::estimator;
int main() {
    State state{Eigen::Vector4d(1,0,0,0),Eigen::Vector3d::Zero(),Matrix6::Identity()};
    auto predicted=AttitudeMEKF::predict(state,Eigen::Matrix3d::Identity(),2.0);
    CHECK((predicted.quaternion-state.quaternion).norm()<1e-14);
    CHECK(std::abs(predicted.covariance(0,0)-(1+4+8.0/3))<1e-12);
    CHECK(std::abs(predicted.covariance(0,3)-(-2-2))<1e-12);
    CHECK(std::abs(predicted.covariance(3,3)-3)<1e-12);
    auto corrected=AttitudeMEKF::correct_gyro(state,Eigen::Vector3d(1,0,0),Eigen::Matrix3d::Identity(),Eigen::Matrix3d::Identity());
    CHECK(std::abs(corrected.state.rate.x()-0.5)<1e-12);
    CHECK(std::abs(corrected.state.covariance(3,3)-0.5)<1e-12);
    CHECK(std::abs(corrected.state.quaternion[0]-1)<1e-12);
    Eigen::Vector4d measured(std::cos(0.1),std::sin(0.1),0,0);
    auto tracker=AttitudeMEKF::correct_star_tracker(state,measured,Eigen::Vector4d(1,0,0,0),Eigen::Matrix3d::Identity());
    CHECK(std::abs(tracker.state.quaternion[1]-std::sin(0.05))<1e-12);
    Sensor sensor; sensor.code=0; sensor.noise_cholesky=Eigen::Matrix3d::Identity();
    Eigen::VectorXd readings(3); readings << 1,0,0;
    auto stepped=AttitudeMEKF::step(state,Eigen::Matrix3d::Zero(),0,readings,{sensor});
    CHECK(std::abs(stepped.rate.x()-0.5)<1e-12);
    readings[0]=std::numeric_limits<double>::quiet_NaN();
    stepped=AttitudeMEKF::step(state,Eigen::Matrix3d::Zero(),0,readings,{sensor});
    CHECK(stepped.rate.norm()==0);
    bool rejected=false;
    try { (void)AttitudeMEKF::predict(state,Eigen::Matrix3d::Zero(),-1); }
    catch (const std::invalid_argument&) { rejected=true; }
    CHECK(rejected);
}
