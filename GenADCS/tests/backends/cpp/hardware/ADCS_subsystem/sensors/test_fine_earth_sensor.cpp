#include "fine_earth_sensor.hpp"
#include "kernels.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>
#include <string>

using omni::hardware::sensors::FineEarthSensor;
namespace {
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+" failed: " #x); } while(false)
using V=FineEarthSensor::V;using Q=FineEarthSensor::Q;using V2=FineEarthSensor::V2;using M2=FineEarthSensor::M2;
const Q identity{{1,0,0,0}};
Q axis_angle(V axis,double angle){double n=std::sqrt(axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2]);double k=std::sin(angle/2)/n;return {{std::cos(angle/2),axis[0]*k,axis[1]*k,axis[2]*k}};}
FineEarthSensor sensor(M2 noise={{{{1e-6,0}},{{0,2e-6}}}},V2 bias={{0,0}},V2 fov={{.6,.7}},Q orientation=identity,Q misalignment=identity){if(noise[0][0]==0)noise={{{{1e-6,0}},{{0,2e-6}}}};return FineEarthSensor(V{{0,0,0}},orientation,misalignment,noise,bias,fov);}
void near(double a,double b,double tolerance=1e-9){CHECK(std::abs(a-b)<=tolerance);}
void test_mounted_angles(){Q om=axis_angle({{0,1,0}},.2),mm=axis_angle({{1,0,0}},-.1),att=axis_angle({{0,0,1}},.3);V earth{{.2,-.1,.97}};double n=std::sqrt(.2*.2+.1*.1+.97*.97);for(double& x:earth)x/=n;FineEarthSensor s=sensor({},{{0,0}},{{.6,.7}},om,mm);Eigen::Quaterniond qm(mm[0],mm[1],mm[2],mm[3]),qo(om[0],om[1],om[2],om[3]),qa(att[0],att[1],att[2],att[3]);Eigen::Vector3d rotated=(qm*qo)*(qa*Eigen::Vector3d(earth[0],earth[1],earth[2]));V2 measured=s.noiseless_biasfree_measurement(earth,att);near(measured[0],std::atan2(rotated[0],rotated[2]));near(measured[1],std::atan2(rotated[1],rotated[2]));}
void test_bias_noise_and_bias_jacobian(){FineEarthSensor s=sensor({{{{1e-8,0}},{{0,2e-8}}}},{{-.01,.02}});V earth{{.1,.2,.97}};V2 base=s.noiseless_biasfree_measurement(earth,identity),biased=s.noiseless_biased_measurement(earth,identity);near(biased[0],base[0]-.01);near(biased[1],base[1]+.02);V2 clean_noise=s.noisy_biasfree_measurement(earth,identity),biased_noise=s.noisy_biased_measurement(earth,identity);for(double x:clean_noise)CHECK(std::isfinite(x));for(double x:biased_noise)CHECK(std::isfinite(x));const M2 bias_jacobian=s.jacobian_bias();for(int i=0;i<2;i++)for(int j=0;j<2;j++)CHECK(bias_jacobian[i][j]==(i==j?1:0));}
void test_attitude_jacobians(){FineEarthSensor s=sensor();V earth{{.1,.2,.97}},zero{{0,0,0}};auto jq=s.jacobian_relative_quaternion(earth,identity,identity),jm=s.jacobian_relative_mrp(earth,identity,zero),ja=s.jacobian_relative_angle_vector(earth,identity,zero);for(int i=0;i<2;i++)for(int j=0;j<3;j++){near(jm[i][j],2*jq[i][j]);near(ja[i][j],.5*jq[i][j]);}Q dq=axis_angle({{1,2,0}},.1);V delta{{.01,-.02,.03}};auto nq=s.jacobian_relative_quaternion(earth,identity,dq),nm=s.jacobian_relative_mrp(earth,identity,delta),na=s.jacobian_relative_angle_vector(earth,identity,delta);for(int i=0;i<2;i++)for(int j=0;j<3;j++){CHECK(std::isfinite(nq[i][j]));CHECK(std::isfinite(nm[i][j]));CHECK(std::isfinite(na[i][j]));}}
void test_availability(){FineEarthSensor s=sensor({},{{0,0}},{{.2,.3}});CHECK(s.availability({{0,0,1}},identity));CHECK(!s.availability({{0,0,1}},identity,false));CHECK(!s.availability({{std::sin(.25),0,std::cos(.25)}},identity));CHECK(!s.availability({{0,std::sin(.4),std::cos(.4)}},identity));}
void test_fisher(){Q mis=axis_angle({{1,0,0}},.1);FineEarthSensor s=sensor({{{{.01,0}},{{0,.02}}}},{{0,0}},{{.6,.7}},identity,mis);V earth{{.1,.2,.97}};V2 a=s.noiseless_biasfree_measurement(earth,identity);auto info=s.fisher_information_matrix(earth,identity);double u=std::tan(a[0]),v=std::tan(a[1]);double k[2][3]={{-u*v/(1+u*u),1,-v/(1+u*u)},{-1,u*v/(1+v*v),u/(1+v*v)}};auto rot=omni::hardware::sensors::kernels::quaternion_matrix(mis);double mounted[2][3]{};for(int i=0;i<2;i++)for(int j=0;j<3;j++)for(int n=0;n<3;n++)mounted[i][j]+=k[i][n]*rot[n][j];for(int i=0;i<3;i++)for(int j=0;j<3;j++)near(info[i][j],mounted[0][i]*mounted[0][j]/.01+mounted[1][i]*mounted[1][j]/.02);auto unavailable=s.fisher_information_matrix(earth,identity,false);for(const auto& row:unavailable)for(double x:row)CHECK(x==0);}
}
int main(){test_mounted_angles();test_bias_noise_and_bias_jacobian();test_attitude_jacobians();test_availability();test_fisher();}
