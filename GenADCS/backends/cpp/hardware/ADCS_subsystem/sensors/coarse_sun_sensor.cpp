#include "coarse_sun_sensor.hpp"
#include "kernels.hpp"
#include "philox.hpp"
#include <cmath>
#include <string>
namespace py=pybind11;
namespace omni::hardware::sensors {
namespace {
template<std::size_t N> std::array<double,N> read(py::handle x,const char* name) {
    auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(x);
    if(!a||a.ndim()!=1||a.shape(0)!=static_cast<py::ssize_t>(N)) throw py::value_error(std::string(name)+" has an invalid shape");
    std::array<double,N> r{};auto v=a.unchecked<1>();for(std::size_t i=0;i<N;++i)r[i]=v(static_cast<py::ssize_t>(i));return r;
}
py::array_t<double> array(const CoarseSunSensor::V& v) { py::array_t<double>a(3);auto o=a.mutable_unchecked<1>();for(int i=0;i<3;++i)o(i)=v[static_cast<std::size_t>(i)];return a; }
}
CoarseSunSensor::CoarseSunSensor(py::handle p,py::handle q,py::handle m,py::handle n,double sens,double sat,double var,double b,double fov)
 : position(read<3>(p,"position")),normal(read<3>(n,"normal")),
   orientation(kernels::normalize(read<4>(q,"orientation"))),misalignment(kernels::normalize(read<4>(m,"misalignment"))),
   sensitivity(sens),saturation_value(sat),noise_variance(var),noise_std(std::sqrt(var)),bias(b),fov_half_angle(fov) {
    double norm=std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
    if(norm==0||sensitivity<=0||saturation_value<=0) throw py::value_error("normal must be nonzero; sensitivity and saturation positive");
    if(noise_variance<=0||fov_half_angle<=0||fov_half_angle>std::acos(-1.0)/2) throw py::value_error("noise variance and FOV must be positive; FOV <= pi/2");
    for(double& x:normal)x/=norm;
}
double CoarseSunSensor::measure(py::handle sun,py::handle attitude,bool wb,bool wn) {
    double z=0; if(wn) { auto draw=omni::helpers::random::standard_normals(seed,random_tick++,0x40000000U,1);z=draw[0]; }
    return kernels::scalar_sun_measurement(read<3>(sun,"sun_inertial"),kernels::read_quaternion(attitude,"attitude"),orientation,misalignment,normal,sensitivity,bias,noise_std,saturation_value,wb,wn,z);
}
double CoarseSunSensor::noiseless_biasfree_measurement(py::handle s,py::handle q)const{return const_cast<CoarseSunSensor*>(this)->measure(s,q,false,false);}
double CoarseSunSensor::noiseless_biased_measurement(py::handle s,py::handle q)const{return const_cast<CoarseSunSensor*>(this)->measure(s,q,true,false);}
double CoarseSunSensor::noisy_biasfree_measurement(py::handle s,py::handle q){return measure(s,q,false,true);}
double CoarseSunSensor::noisy_biased_measurement(py::handle s,py::handle q){return measure(s,q,true,true);}
double CoarseSunSensor::jacobian_bias()const{return 1.0;}
py::array_t<double> CoarseSunSensor::jacobian_relative_quaternion(py::handle s,py::handle q,py::handle d)const{return array(kernels::coarse_sun_attitude_jacobian(read<3>(s,"sun_inertial"),kernels::read_quaternion(q,"attitude"),orientation,misalignment,normal,sensitivity,kernels::read_quaternion(d,"delta_quaternion"),0));}
py::array_t<double> CoarseSunSensor::jacobian_relative_mrp(py::handle s,py::handle q,py::handle d)const{return array(kernels::coarse_sun_attitude_jacobian(read<3>(s,"sun_inertial"),kernels::read_quaternion(q,"attitude"),orientation,misalignment,normal,sensitivity,kernels::read_vector3(d,"delta_mrp"),1));}
py::array_t<double> CoarseSunSensor::jacobian_relative_angle_vector(py::handle s,py::handle q,py::handle d)const{return array(kernels::coarse_sun_attitude_jacobian(read<3>(s,"sun_inertial"),kernels::read_quaternion(q,"attitude"),orientation,misalignment,normal,sensitivity,kernels::read_vector3(d,"delta_angle"),2));}
bool CoarseSunSensor::availability(py::handle s,py::handle q,bool visible)const{return kernels::coarse_sun_available(read<3>(s,"sun_inertial"),kernels::read_quaternion(q,"attitude"),orientation,misalignment,normal,fov_half_angle,enabled,visible);}
py::array_t<double> CoarseSunSensor::fisher_information_matrix(py::handle s,py::handle q,bool visible)const {
    using M=std::array<std::array<double,3>,3>; M info{};
    if(availability(s,q,visible)) info=kernels::coarse_sun_information(read<3>(s,"sun_inertial"),kernels::read_quaternion(q,"attitude"),orientation,misalignment,normal,sensitivity,noise_variance);
    py::array_t<double> out({3,3});auto v=out.mutable_unchecked<2>();for(int i=0;i<3;++i)for(int j=0;j<3;++j)v(i,j)=info[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];return out;
}
}
