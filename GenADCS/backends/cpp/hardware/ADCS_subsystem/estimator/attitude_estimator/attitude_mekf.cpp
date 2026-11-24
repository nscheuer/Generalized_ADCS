#include "attitude_mekf.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

namespace omni::hardware::estimator {
namespace {
using Q = Eigen::Quaterniond;
Q quat(const Eigen::Vector4d& v) { return Q(v[0], v[1], v[2], v[3]); }
Eigen::Vector4d vec(Q q) { q.normalize(); return {q.w(), q.x(), q.y(), q.z()}; }
Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d s;
    s << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return s;
}
Q expmap(const Eigen::Vector3d& v) {
    double angle = v.norm();
    if (angle < 1e-12) return Q(1, 0.5*v.x(), 0.5*v.y(), 0.5*v.z()).normalized();
    return Q(Eigen::AngleAxisd(angle, v/angle));
}
Eigen::Vector3d logmap(Q q) {
    q.normalize();
    if (q.w() < 0) q.coeffs() *= -1;
    const double n = q.vec().norm();
    if (n < 1e-12) return 2*q.vec();
    return 2*std::atan2(n, q.w())*q.vec()/n;
}
Eigen::Matrix3d left_jacobian(const Eigen::Vector3d& v) {
    const double angle = v.norm();
    const Eigen::Matrix3d k = skew(v);
    if (angle < 1e-6) return Eigen::Matrix3d::Identity() + 0.5*k + k*k/6.0;
    return Eigen::Matrix3d::Identity() + ((1-std::cos(angle))/(angle*angle))*k +
           ((angle-std::sin(angle))/(angle*angle*angle))*k*k;
}
}
State AttitudeMEKF::predict(const State& state, const Eigen::Matrix3d& a, double dt) {
    if (!std::isfinite(dt) || dt < 0) throw std::invalid_argument("dt must be finite and nonnegative");
    const Eigen::Vector3d rotation = -state.rate*dt;
    const Q increment = expmap(rotation);
    Matrix6 phi = Matrix6::Identity(), noise = Matrix6::Zero();
    phi.topLeftCorner<3,3>() = increment.toRotationMatrix();
    phi.topRightCorner<3,3>() = -left_jacobian(rotation)*dt;
    noise.topLeftCorner<3,3>() = a*(dt*dt*dt/3.0);
    noise.topRightCorner<3,3>() = -a*(dt*dt/2.0);
    noise.bottomLeftCorner<3,3>() = noise.topRightCorner<3,3>();
    noise.bottomRightCorner<3,3>() = a*dt;
    Matrix6 p = phi*state.covariance*phi.transpose() + noise;
    return {vec(increment*quat(state.quaternion)), state.rate, 0.5*(p+p.transpose())};
}
Correction AttitudeMEKF::correct(const State& state, const Eigen::Vector3d& residual,
                                  const Eigen::Matrix<double,3,6>& h, const Eigen::Matrix3d& r) {
    const Eigen::Matrix3d s = h*state.covariance*h.transpose()+r;
    const auto solver = s.ldlt();
    if (solver.info() != Eigen::Success || (solver.vectorD().array().abs() < 1e-15).any())
        throw std::invalid_argument("innovation covariance is singular");
    const Eigen::Matrix<double,6,3> gain = solver.solve((h*state.covariance).eval()).transpose();
    const Vector6 delta = gain*residual;
    const Matrix6 remainder = Matrix6::Identity()-gain*h;
    Matrix6 p = remainder*state.covariance*remainder.transpose()+gain*r*gain.transpose();
    Matrix6 reset = Matrix6::Identity();
    reset.topLeftCorner<3,3>() += 0.5*skew(delta.head<3>());
    p = reset*p*reset.transpose();
    State result{vec(expmap(delta.head<3>())*quat(state.quaternion)), state.rate+delta.tail<3>(), 0.5*(p+p.transpose())};
    return {result, residual, s, gain, residual-h*delta};
}
Correction AttitudeMEKF::correct_gyro(const State& state, const Eigen::Vector3d& measured_rate,
                                       const Eigen::Matrix3d& mounting, const Eigen::Matrix3d& covariance) {
    Eigen::Matrix<double,3,6> h = Eigen::Matrix<double,3,6>::Zero();
    h.rightCols<3>() = mounting;
    return correct(state, measured_rate-mounting*state.rate, h, covariance);
}
Correction AttitudeMEKF::correct_star_tracker(const State& state, const Eigen::Vector4d& measured_quaternion,
                                               const Eigen::Vector4d& mounting_quaternion, const Eigen::Matrix3d& covariance) {
    const Q mount = quat(mounting_quaternion).normalized();
    const Q predicted = mount*quat(state.quaternion);
    Eigen::Matrix<double,3,6> h = Eigen::Matrix<double,3,6>::Zero();
    h.leftCols<3>() = mount.toRotationMatrix();
    return correct(state, logmap(quat(measured_quaternion)*predicted.conjugate()), h, covariance);
}
State AttitudeMEKF::step(const State& state, const Eigen::Matrix3d& a, double dt,
                         const Eigen::VectorXd& readings, const std::vector<Sensor>& sensors) {
    State current = predict(state, a, dt);
    for (const Sensor& sensor : sensors) {
        if (!sensor.enabled || (sensor.code != 0 && sensor.code != 2)) continue;
        const Eigen::Index width = sensor.code == 0 ? 3 : 4;
        if (sensor.offset < 0 || sensor.offset + width > readings.size())
            throw std::invalid_argument("sensor reading offset is out of range");
        const Eigen::VectorXd value = readings.segment(sensor.offset, width);
        if (!value.allFinite()) continue;
        const Q mount = (quat(sensor.misalignment)*quat(sensor.orientation)).normalized();
        const Eigen::Matrix3d noise = sensor.noise_cholesky*sensor.noise_cholesky.transpose();
        if (sensor.code == 0)
            current = correct_gyro(current, value.head<3>()-sensor.bias, mount.toRotationMatrix(), noise).state;
        else
            current = correct_star_tracker(current, value.head<4>(), vec(mount), noise).state;
    }
    return current;
}
}
