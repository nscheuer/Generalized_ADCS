#include "accelerometer.hpp"
#include "kernels.hpp"
#include "philox.hpp"
#include <cmath>
#include <string>
namespace py=pybind11;
namespace omni::hardware::sensors {
namespace {
template<std::size_t N> std::array<double,N> read_vector(py::handle x,const char* name) {
    auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(x);
    if(!a||a.ndim()!=1||a.shape(0)!=static_cast<py::ssize_t>(N)) throw py::value_error(std::string(name)+" has an invalid shape");
    std::array<double,N> r{}; auto v=a.unchecked<1>();
    for (std::size_t i = 0; i < N; ++i) r[i] = v(static_cast<py::ssize_t>(i));
    return r;
}
}
Accelerometer::Accelerometer(py::handle p,py::handle q,py::handle m,double limit,py::handle n,py::handle b)
 : position(kernels::read_vector3(p,"position")),orientation(kernels::normalize(kernels::read_quaternion(q,"orientation"))),
   misalignment(kernels::normalize(kernels::read_quaternion(m,"misalignment"))),saturation_value(limit),
   noise(kernels::read_matrix(n,"noise")),noise_cholesky(kernels::cholesky(noise)),bias(kernels::read_vector3(b,"bias")) {
    if(!(saturation_value>0.0)) throw py::value_error("saturation_value must be positive");
}
Accelerometer::V Accelerometer::specific_force(py::handle a,py::handle g) const {
    V acceleration=read_vector<3>(a,"acceleration_inertial"), gravity=read_vector<3>(g,"gravity_inertial");
    for (std::size_t i = 0; i < 3; ++i) acceleration[i] -= gravity[i];
    return acceleration;
}
py::array_t<double> Accelerometer::measure(py::handle a,py::handle g,py::handle q,bool wb,bool wn) {
    V draw{}; if(wn) { auto z=omni::helpers::random::standard_normals(seed,random_tick++,0x40000000U,3); for(std::size_t i=0;i<3;++i) draw[i]=z[i]; }
    return kernels::array_vector(kernels::accelerometer_measurement(read_vector<3>(a,"acceleration_inertial"),
        read_vector<3>(g,"gravity_inertial"),kernels::read_quaternion(q,"attitude"),orientation,misalignment,
        bias,noise_cholesky,saturation_value,wb,wn,wn?&draw:nullptr));
}
py::array_t<double> Accelerometer::noiseless_biasfree_measurement(py::handle a,py::handle g,py::handle q) const { return const_cast<Accelerometer*>(this)->measure(a,g,q,false,false); }
py::array_t<double> Accelerometer::noiseless_biased_measurement(py::handle a,py::handle g,py::handle q) const { return const_cast<Accelerometer*>(this)->measure(a,g,q,true,false); }
py::array_t<double> Accelerometer::noisy_biasfree_measurement(py::handle a,py::handle g,py::handle q) { return measure(a,g,q,false,true); }
py::array_t<double> Accelerometer::noisy_biased_measurement(py::handle a,py::handle g,py::handle q) { return measure(a,g,q,true,true); }
py::array_t<double> Accelerometer::jacobian_relative_quaternion(py::handle a,py::handle g,py::handle q,py::handle d) const {
    return kernels::array_matrix(kernels::attitude_vector_jacobian_quaternion(kernels::rotate_inertial_to_sensor(specific_force(a,g),kernels::read_quaternion(q,"attitude"),orientation,misalignment),kernels::read_quaternion(d,"delta_quaternion")));
}
py::array_t<double> Accelerometer::jacobian_relative_mrp(py::handle a,py::handle g,py::handle q,py::handle d) const {
    return kernels::array_matrix(kernels::attitude_vector_jacobian_mrp(kernels::rotate_inertial_to_sensor(specific_force(a,g),kernels::read_quaternion(q,"attitude"),orientation,misalignment),kernels::read_vector3(d,"delta_mrp")));
}
py::array_t<double> Accelerometer::jacobian_relative_angle_vector(py::handle a,py::handle g,py::handle q,py::handle d) const {
    return kernels::array_matrix(kernels::attitude_vector_jacobian_angle(kernels::rotate_inertial_to_sensor(specific_force(a,g),kernels::read_quaternion(q,"attitude"),orientation,misalignment),kernels::read_vector3(d,"delta_angle")));
}
py::array_t<double> Accelerometer::jacobian_bias() const { return kernels::array_matrix(kernels::identity()); }
bool Accelerometer::availability() const { return enabled; }
}
