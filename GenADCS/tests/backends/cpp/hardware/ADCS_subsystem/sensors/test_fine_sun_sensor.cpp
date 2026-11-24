#include "kernels.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

using namespace omni::hardware::sensors::kernels;
namespace {
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #x); } while(false)
using V=std::array<double,3>; using Q=std::array<double,4>; using V2=std::array<double,2>; using C2=std::array<std::array<double,2>,2>;
Q axis_angle(V axis,double angle){double n=std::sqrt(axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2]);double s=std::sin(angle/2)/n;return {{std::cos(angle/2),axis[0]*s,axis[1]*s,axis[2]*s}};}
V rotate(Q q,V x){return multiply(quaternion_matrix(q),x);}
V2 angles(V s,Q q,Q o,Q m){return fine_direction_measurement(s,q,o,m,{{0,0}},C2{},false,false,nullptr);}
void check_close(double a,double b,double tol=1e-8){CHECK(std::abs(a-b)<=tol);}
void test_mounted_angles(){Q om=axis_angle({{0,1,0}},.2),mm=axis_angle({{1,0,0}},-.1),att=axis_angle({{0,0,1}},.3);V sun{{.2,-.1,.97}};double n=std::sqrt(.2*.2+.1*.1+.97*.97);for(double&x:sun)x/=n;auto actual=angles(sun,att,om,mm);V expected_vector=rotate(multiply(mm,om),rotate(att,sun));check_close(actual[0],std::atan2(expected_vector[0],expected_vector[2]));check_close(actual[1],std::atan2(expected_vector[1],expected_vector[2]));}
void test_bias_and_noise(){Q id{{1,0,0,0}};C2 l{{{{1e-4,0}},{{0,std::sqrt(2e-8)}}}};V sun{{.1,.2,.97}};auto base=fine_direction_measurement(sun,id,id,id,{{0,0}},l,false,false,nullptr);auto biased=fine_direction_measurement(sun,id,id,id,{{.01,-.02}},l,true,false,nullptr);check_close(biased[0],base[0]+.01);check_close(biased[1],base[1]-.02);V2 z{{0,0}};auto noisy=fine_direction_measurement(sun,id,id,id,{{0,0}},l,false,true,&z);check_close(noisy[0],base[0]);check_close(noisy[1],base[1]);}
void test_jacobians(){Q id{{1,0,0,0}};V sun{{.1,.2,.97}},mrp{{0,0,0}};auto jq=fine_sun_attitude_jacobian(sun,id,id,id,id,0);auto jm=fine_sun_attitude_jacobian(sun,id,id,id,mrp,1);auto ja=fine_sun_attitude_jacobian(sun,id,id,id,mrp,2);for(int i=0;i<2;i++)for(int j=0;j<3;j++){check_close(jm[i][j],2*jq[i][j]);check_close(ja[i][j],.5*jq[i][j]);}V delta{{.04,-.03,.02}};auto analytic=fine_sun_attitude_jacobian(sun,id,id,id,delta,2);double h=1e-6;for(int j=0;j<3;j++){V plus=delta,minus=delta;plus[j]+=h;minus[j]-=h;auto numeric_p=rotate(axis_angle(plus,std::sqrt(plus[0]*plus[0]+plus[1]*plus[1]+plus[2]*plus[2])),sun);auto numeric_m=rotate(axis_angle(minus,std::sqrt(minus[0]*minus[0]+minus[1]*minus[1]+minus[2]*minus[2])),sun);auto ap=angles(numeric_p,id,id,id),am=angles(numeric_m,id,id,id);for(int i=0;i<2;i++)check_close(analytic[i][j],(ap[i]-am[i])/(2*h),1e-5);}}
void test_availability(){Q id{{1,0,0,0}};std::array<double,2> fov{{.2,.3}};CHECK(fine_direction_available({{0,0,1}},id,id,id,fov,true,true));CHECK(!fine_direction_available({{0,0,1}},id,id,id,fov,true,false));CHECK(!fine_direction_available({{std::sin(.25),0,std::cos(.25)}},id,id,id,fov,true,true));CHECK(!fine_direction_available({{0,std::sin(.4),std::cos(.4)}},id,id,id,fov,true,true));}
void test_fisher(){Q id{{1,0,0,0}};V sun{{.1,.2,.97}};V2 a=angles(sun,id,id,id);C2 cov{{{{.01,0}},{{0,.02}}}};M info=fine_sun_fisher_information(a,cov,id,id);double u=std::tan(a[0]),v=std::tan(a[1]);double k[2][3]={{-u*v/(1+u*u),1,-v/(1+u*u)},{-1,u*v/(1+v*v),u/(1+v*v)}};for(int i=0;i<3;i++)for(int j=0;j<3;j++)check_close(info[i][j],k[0][i]*k[0][j]/.01+k[1][i]*k[1][j]/.02);}
}
int main(){test_mounted_angles();test_bias_and_noise();test_jacobians();test_availability();test_fisher();}
