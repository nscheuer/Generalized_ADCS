#include "autonomous_star_tracker.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

using omni::hardware::sensors::AutonomousStarTracker;
#define CHECK(c) do { if(!(c)) throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+" failed: "#c); } while(false)
namespace {
const AutonomousStarTracker::Q id{{1,0,0,0}};
AutonomousStarTracker make(AutonomousStarTracker::M noise={{{1e-8,0,0},{0,1e-8,0},{0,0,1e-8}}},
 AutonomousStarTracker::Q o=id, AutonomousStarTracker::Q m=id) {
 return {{{0,0,0}},o,m,1.,noise,{{0,0,0}}};
}
AutonomousStarTracker::Q axis(AutonomousStarTracker::V v,double a){double n=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);return {{std::cos(a/2),std::sin(a/2)*v[0]/n,std::sin(a/2)*v[1]/n,std::sin(a/2)*v[2]/n}};}
AutonomousStarTracker::Q product(const AutonomousStarTracker::Q&a,const AutonomousStarTracker::Q&b){return {{a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]}};}
AutonomousStarTracker::M rotation(const AutonomousStarTracker::Q&q){const auto [w,x,y,z]=q;return {{{1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)},{2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)},{2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)}}};}
AutonomousStarTracker::M matmul(const AutonomousStarTracker::M&a,const AutonomousStarTracker::M&b){AutonomousStarTracker::M r{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)r[i][j]+=a[i][k]*b[k][j];return r;}
AutonomousStarTracker::M transpose(const AutonomousStarTracker::M&a){AutonomousStarTracker::M r{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)r[i][j]=a[j][i];return r;}
void availability(){auto s=make(); s.fov_half_angle=.2; CHECK(s.availability());CHECK(s.availability(id,AutonomousStarTracker::V{{1,0,0}})); CHECK(!s.availability(id,AutonomousStarTracker::V{{0,0,1}}));s.enabled=false;CHECK(!s.availability());}
void mounted_measurement(){auto s=make({{{1e-8,0,0},{0,1e-8,0},{0,0,1e-8}}},axis({{0,0,1}},.3),axis({{1,0,0}},-.2));const auto q=axis({{0,1,0}},.5), actual=s.noiseless_biasfree_measurement(q);const auto expected=product(s.misalignment,product(s.orientation,q));for(int i=0;i<4;++i)CHECK(std::abs(actual[i]-expected[i])<1e-14);}
void noise_residual_jacobians(){auto tiny=make({{{1e-12,0,0},{0,1e-12,0},{0,0,1e-12}}});const auto q=axis({{1,2,-1}},.7),measured=tiny.noisy_biasfree_measurement(q);double norm=0;for(double x:measured){CHECK(std::isfinite(x));norm+=x*x;}CHECK(std::abs(norm-1)<1e-14);double distance=0,opposite=0;for(int i=0;i<4;++i){distance+=(measured[i]-q[i])*(measured[i]-q[i]);opposite+=(measured[i]+q[i])*(measured[i]+q[i]);}CHECK(std::sqrt(std::min(distance,opposite))<1e-5);
 auto s=make();const auto identity_result=s.noiseless_biased_measurement(id), noisy=s.noisy_biased_measurement(id);for(int i=0;i<4;++i){CHECK(identity_result[i]==id[i]);CHECK(std::isfinite(noisy[i]));}norm=0;for(double x:noisy)norm+=x*x;CHECK(std::abs(norm-1)<1e-14);
 const auto zero=s.residual(q,q), neg=s.residual({{-q[0],-q[1],-q[2],-q[3]}},q);for(int i=0;i<3;++i){CHECK(std::abs(zero[i])<1e-14);CHECK(std::abs(neg[i])<1e-14);}const auto delta=AutonomousStarTracker::V{{2e-4,-3e-4,1e-4}};const double angle=std::sqrt(delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]);const auto small=s.residual(axis({{delta[0]/angle,delta[1]/angle,delta[2]/angle}},angle),id);for(int i=0;i<3;++i)CHECK(std::abs(small[i]-delta[i])<1e-11);
 const auto jq=s.jacobian_relative_quaternion(),jm=s.jacobian_relative_mrp({{0,0,0}}),ja=s.jacobian_relative_angle_vector({{0,0,0}});for(int i=0;i<3;++i)for(int j=0;j<3;++j){CHECK(jq[i][j]==(i==j?2:0));CHECK(jm[i][j]==(i==j?4:0));CHECK(std::abs(ja[i][j]-(i==j))<1e-15);}const auto away=s.jacobian_relative_angle_vector({{.1,-.2,.05}});for(const auto&row:away)for(double value:row)CHECK(std::isfinite(value));
 const AutonomousStarTracker::V parameters{{.12,-.08,.04}};const double step=1e-6;for(int col=0;col<3;++col){auto plus=parameters,minus=parameters;plus[col]+=step;minus[col]-=step;const double sp=plus[0]*plus[0]+plus[1]*plus[1]+plus[2]*plus[2],sm=minus[0]*minus[0]+minus[1]*minus[1]+minus[2]*minus[2];for(int row=0;row<3;++row){const double fp=4*plus[row]/(1+sp),fm=4*minus[row]/(1+sm);const double finite_difference=(fp-fm)/(2*step);CHECK(std::abs(s.jacobian_relative_mrp(parameters)[row][col]-finite_difference)<1e-9);}}
}
void information_and_invalid(){const auto covariance=AutonomousStarTracker::M{{{.2,0,0},{0,.3,0},{0,0,.5}}};const auto orientation=axis({{0,0,1}},.4),misalignment=axis({{1,0,0}},-.2);auto s=make(covariance,orientation,misalignment);const auto r=rotation(product(misalignment,orientation));const AutonomousStarTracker::M inverse_cov{{{5,0,0},{0,1/.3,0},{0,0,2}}};const auto expected=matmul(transpose(r),matmul(inverse_cov,r)),actual=s.fisher_information_matrix();for(int i=0;i<3;++i)for(int j=0;j<3;++j)CHECK(std::abs(actual[i][j]-expected[i][j])<1e-12);CHECK(s.availability());auto disabled=make();disabled.enabled=false;for(const auto&r0:disabled.fisher_information_matrix())for(double x:r0)CHECK(x==0);bool threw=false;try{AutonomousStarTracker invalid({{0,0,0}},id,id,1,{{{1,0,0},{0,1,0},{0,0,1}}},{{0,0,0}},0);}catch(const std::invalid_argument&){threw=true;}CHECK(threw);}
}
int main(){availability();mounted_measurement();noise_residual_jacobians();information_and_invalid();}
