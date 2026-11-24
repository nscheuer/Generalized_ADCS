#ifndef GENADCS_HARDWARE_SENSORS_FINE_SUN_SENSOR_HPP
#define GENADCS_HARDWARE_SENSORS_FINE_SUN_SENSOR_HPP
#include <pybind11/numpy.h>
#include <array>
#include <cmath>
#include <stdexcept>
#include "kernels.hpp"
#include "philox.hpp"
namespace omni::hardware::sensors {
struct FineSunSensor {
 using V=std::array<double,3>;using Q=std::array<double,4>;using V2=std::array<double,2>;using M2=std::array<std::array<double,2>,2>;using M23=std::array<std::array<double,3>,2>;
 FineSunSensor(pybind11::handle,pybind11::handle,pybind11::handle,pybind11::handle,pybind11::handle,pybind11::handle);
 FineSunSensor(V p,Q o,Q m,M2 covariance,V2 b,V2 fov)
     :position(p),orientation(kernels::normalize(o)),misalignment(kernels::normalize(m)),noise(covariance),bias(b),fov_half_angles(fov){
     const double l00=std::sqrt(noise[0][0]);
     const double l10=noise[1][0]/l00;
     const double remainder=noise[1][1]-l10*l10;
     if(!(noise[0][0]>0)||!(remainder>0))throw std::invalid_argument("noise covariance must be positive definite");
     noise_cholesky={{{{l00,0}},{{l10,std::sqrt(remainder)}}}};
 }
 V2 native_measurement(const V& direction,const Q& attitude,bool with_bias,bool with_noise){
     V2 draw{};
     if(with_noise){const std::vector<double> z=omni::helpers::random::standard_normals(seed,random_tick++,0x40000000U,2);draw={{z[0],z[1]}};}
     return kernels::fine_direction_measurement(direction,attitude,orientation,misalignment,bias,noise_cholesky,with_bias,with_noise,with_noise?&draw:nullptr);
 }
 M23 native_jacobian_quaternion(const V& direction,const Q& attitude,const Q& delta)const{return kernels::fine_sun_attitude_jacobian(direction,attitude,orientation,misalignment,delta,0);}
 M2 native_jacobian_bias()const{return {{{{1,0}},{{0,1}}}};}
 M23 native_jacobian_mrp(const V& direction,const Q& attitude,const V& delta)const{return kernels::fine_sun_attitude_jacobian(direction,attitude,orientation,misalignment,delta,1);}
 M23 native_jacobian_angle(const V& direction,const Q& attitude,const V& delta)const{return kernels::fine_sun_attitude_jacobian(direction,attitude,orientation,misalignment,delta,2);}
 bool native_availability(const V& direction,const Q& attitude,bool visible=true)const{return kernels::fine_direction_available(direction,attitude,orientation,misalignment,fov_half_angles,enabled,visible);}
 std::array<std::array<double,3>,3> native_fisher_information(const V& direction,const Q& attitude,bool visible=true)const{
     if(!native_availability(direction,attitude,visible))return {};
     const V s=kernels::rotate_inertial_to_sensor(direction,attitude,orientation,misalignment);
     return kernels::fine_sun_fisher_information({{std::atan2(s[0],s[2]),std::atan2(s[1],s[2])}},noise,orientation,misalignment);
 }
 pybind11::array_t<double> noiseless_biasfree_measurement(pybind11::handle,pybind11::handle)const;
 pybind11::array_t<double> noiseless_biased_measurement(pybind11::handle,pybind11::handle)const;
 pybind11::array_t<double> noisy_biasfree_measurement(pybind11::handle,pybind11::handle);
 pybind11::array_t<double> noisy_biased_measurement(pybind11::handle,pybind11::handle);
 pybind11::array_t<double> jacobian_bias()const;
 pybind11::array_t<double> jacobian_relative_quaternion(pybind11::handle,pybind11::handle,pybind11::handle)const;
 pybind11::array_t<double> jacobian_relative_mrp(pybind11::handle,pybind11::handle,pybind11::handle)const;
 pybind11::array_t<double> jacobian_relative_angle_vector(pybind11::handle,pybind11::handle,pybind11::handle)const;
 bool availability(pybind11::handle,pybind11::handle,bool=true)const;
 pybind11::array_t<double> fisher_information_matrix(pybind11::handle,pybind11::handle,bool=true)const;
 V position{};Q orientation{},misalignment{};M2 noise{},noise_cholesky{};V2 bias{},fov_half_angles{};bool enabled=true;unsigned long long seed=0,random_tick=0;
private: pybind11::array_t<double> measure(pybind11::handle,pybind11::handle,bool,bool);
};}
#endif
