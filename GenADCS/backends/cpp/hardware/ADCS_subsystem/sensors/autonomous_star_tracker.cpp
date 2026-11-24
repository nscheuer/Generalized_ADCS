#include "autonomous_star_tracker.hpp"
#include "sensor_kernels.hpp"
#include "philox.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace omni::hardware::sensors {
namespace {
AutonomousStarTracker::Q conjugate(const AutonomousStarTracker::Q& q) {
    return {{q[0], -q[1], -q[2], -q[3]}};
}
}

AutonomousStarTracker::AutonomousStarTracker(V p, Q o, Q m, double saturation, M covariance, V b, double fov)
    : position(p), bias(b), orientation(kernels::normalize(o)), misalignment(kernels::normalize(m)),
      noise(covariance), noise_cholesky(kernels::cholesky(covariance)), saturation_value(saturation), fov_half_angle(fov) {
    if (!(saturation_value > 0.0) || !std::isfinite(saturation_value)) throw std::invalid_argument("saturation_value must be positive");
    if (!std::isfinite(fov_half_angle) || fov_half_angle <= 0.0 || fov_half_angle > std::acos(-1.0))
        throw std::invalid_argument("fov_half_angle must be in (0, pi]");
}
AutonomousStarTracker::Q AutonomousStarTracker::noiseless_biasfree_measurement(const Q& attitude) const {
    return kernels::normalize(kernels::multiply(misalignment, kernels::multiply(orientation, attitude)));
}
AutonomousStarTracker::Q AutonomousStarTracker::noiseless_biased_measurement(const Q& q) const { return noiseless_biasfree_measurement(q); }
AutonomousStarTracker::Q AutonomousStarTracker::noisy_biasfree_measurement(const Q& attitude) {
    const auto draws = omni::helpers::random::standard_normals(seed, random_tick++, 0x40000000U, 3);
    V eta{};
    for(int i=0;i<3;++i) for(int j=0;j<=i;++j) eta[i] += noise_cholesky[i][j] * draws[j];
    const double norm=std::sqrt(1.0+0.25*(eta[0]*eta[0]+eta[1]*eta[1]+eta[2]*eta[2]));
    const Q nq{{1.0/norm, .5*eta[0]/norm, .5*eta[1]/norm, .5*eta[2]/norm}};
    return kernels::normalize(kernels::multiply(nq, noiseless_biasfree_measurement(attitude)));
}
AutonomousStarTracker::Q AutonomousStarTracker::noisy_biased_measurement(const Q& q) { return noisy_biasfree_measurement(q); }
AutonomousStarTracker::V AutonomousStarTracker::residual(const Q& measured, const Q& estimated) const {
    Q relative = kernels::normalize(kernels::multiply(measured, conjugate(estimated)));
    if (relative[0] < 0.0) for(double& value:relative)value=-value;
    return {{2*relative[1],2*relative[2],2*relative[3]}};
}
AutonomousStarTracker::M AutonomousStarTracker::jacobian_relative_quaternion() const {
    auto r=kernels::identity(); for(auto& row:r)for(double& x:row)x*=2.0; return r;
}
AutonomousStarTracker::M AutonomousStarTracker::jacobian_relative_mrp(const V& d) const {
    const double s=d[0]*d[0]+d[1]*d[1]+d[2]*d[2], factor=4.0/((1+s)*(1+s));
    M r{}; for(int i=0;i<3;++i)for(int j=0;j<3;++j)r[i][j]=factor*((i==j?1+s:0)-2*d[i]*d[j]); return r;
}
AutonomousStarTracker::M AutonomousStarTracker::jacobian_relative_angle_vector(const V& d) const {
    const double a=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    double x,y;
    if(a<1e-7){x=1-a*a/24;y=-1.0/12+a*a/480;}
    else{x=2*std::sin(a/2)/a;y=std::cos(a/2)/(a*a)-2*std::sin(a/2)/(a*a*a);}
    M r{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)r[i][j]=x*(i==j)+y*d[i]*d[j];return r;
}
bool AutonomousStarTracker::availability(std::optional<Q> attitude, std::optional<V> earth, std::optional<V> sun) const {
    if (!enabled) return false;
    if (!attitude) return true;
    for(const auto* direction : {&earth,&sun}) if(*direction) {
        const V v=kernels::rotate_inertial_to_sensor(**direction,*attitude,orientation,misalignment);
        if(v[2]>0 && std::atan2(std::hypot(v[0],v[1]),v[2])<fov_half_angle)return false;
    }
    return true;
}
AutonomousStarTracker::M AutonomousStarTracker::fisher_information_matrix() const {
    if (!enabled) return {};
    const M rotation = kernels::quaternion_matrix(kernels::multiply(misalignment, orientation));
    return kernels::multiply(kernels::transpose(rotation),
        kernels::multiply(kernels::inverse_from_cholesky(noise_cholesky), rotation));
}
}
