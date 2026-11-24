#include "../check.hpp"
#include <pybind11/numpy.h>
#include <cmath>
#include <initializer_list>

namespace {
py::array_t<double> vec(std::initializer_list<double> values) {
    py::array_t<double> a(static_cast<py::ssize_t>(values.size())); auto out=a.mutable_unchecked<1>();
    py::ssize_t i=0; for(double x:values) out(i++)=x; return a;
}
py::array_t<double> mat(double d=1e-6) {
    py::array_t<double> a({3,3}); auto out=a.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < 3; ++i)
        for (py::ssize_t j = 0; j < 3; ++j) out(i,j) = i == j ? d : 0.;
    return a;
}
py::object make_sensor(py::object type, double limit=100., py::object bias=py::none(),
                       py::object orientation=py::none(),py::object misalignment=py::none()) {
    if (bias.is_none()) bias=vec({0,0,0});
    if (orientation.is_none()) orientation=vec({1,0,0,0});
    if(misalignment.is_none()) misalignment=vec({1,0,0,0});
    return type(vec({0,0,0}),orientation,misalignment,limit,mat(),bias);
}
void close_vec(py::handle actual,const double* expected,double tol=1e-11) {
    auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(actual); CHECK(a&&a.ndim()==1&&a.shape(0)==3);
    auto v=a.unchecked<1>(); for(int i=0;i<3;++i) CHECK(std::abs(v(i)-expected[i])<tol);
}
void finite_matrix(py::handle actual) {
    auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(actual); CHECK(a&&a.ndim()==2&&a.shape(0)==3&&a.shape(1)==3);
    auto v=a.unchecked<2>(); for(int i=0;i<3;++i) for(int j=0;j<3;++j) CHECK(std::isfinite(v(i,j)));
}
void test_specific_force_and_mounting(py::object type) {
    py::object orientation=vec({std::cos(.25),0,0,std::sin(.25)});
    py::object misalignment=vec({std::cos(-.1),std::sin(-.1),0,0});
    py::object attitude=vec({std::cos(.15),0,std::sin(.15),0});
    py::object sensor=make_sensor(type,100.,py::none(),orientation,misalignment);
    py::object out=sensor.attr("noiseless_biasfree_measurement")(vec({2,3,4}),vec({.5,1,-1}),attitude);
    // Independent expected value: body rotation about y, then z mounting, then x misalignment.
    const double body[]={std::cos(.3)*1.5+std::sin(.3)*5.,2.,-std::sin(.3)*1.5+std::cos(.3)*5.};
    const double z[]={std::cos(.5)*body[0]-std::sin(.5)*body[1],std::sin(.5)*body[0]+std::cos(.5)*body[1],body[2]};
    const double expected[]={z[0],std::cos(-.2)*z[1]-std::sin(-.2)*z[2],std::sin(-.2)*z[1]+std::cos(-.2)*z[2]};
    close_vec(out,expected);
}
void test_bias_noise_and_jacobians(py::object type) {
    py::object sensor=make_sensor(type,100.,vec({.1,-.2,.3}));
    py::object a=vec({2,3,4}),g=vec({1,1,1}),q=vec({1,0,0,0});
    const double base[]={1,2,3},biased[]={1.1,1.8,3.3};
    close_vec(sensor.attr("noiseless_biasfree_measurement")(a,g,q),base);
    close_vec(sensor.attr("noiseless_biased_measurement")(a,g,q),biased);
    for(const char* name:{"noisy_biasfree_measurement","noisy_biased_measurement"}) {
        auto result=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(sensor.attr(name)(a,g,q));
        CHECK(result&&result.ndim()==1&&result.shape(0)==3); auto v=result.unchecked<1>(); for(int i=0;i<3;++i) CHECK(std::isfinite(v(i)));
    }
    auto jq=sensor.attr("jacobian_relative_quaternion")(a,g,q,q);
    auto jm=sensor.attr("jacobian_relative_mrp")(a,g,q,vec({0,0,0}));
    auto ja=sensor.attr("jacobian_relative_angle_vector")(a,g,q,vec({0,0,0}));
    finite_matrix(jq); finite_matrix(jm); finite_matrix(ja);
    py::array_t<double,py::array::c_style|py::array::forcecast> qmat=jq;
    py::array_t<double,py::array::c_style|py::array::forcecast> mmat=jm;
    py::array_t<double,py::array::c_style|py::array::forcecast> amat=ja;
    auto qv=qmat.unchecked<2>(); auto mv=mmat.unchecked<2>(); auto av=amat.unchecked<2>();
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) { CHECK(std::abs(mv(i,j)-2*qv(i,j))<1e-12); CHECK(std::abs(av(i,j)-.5*qv(i,j))<1e-12); }
    auto bias_jacobian=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(sensor.attr("jacobian_bias")());
    CHECK(bias_jacobian&&bias_jacobian.ndim()==2&&bias_jacobian.shape(0)==3&&bias_jacobian.shape(1)==3);
    auto bv=bias_jacobian.unchecked<2>();
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) CHECK(bv(i,j)==(i==j?1.0:0.0));
}
void test_nonzero_attitude_jacobians_are_finite(py::object type) {
    py::object sensor=make_sensor(type), a=vec({1,1,1}),g=vec({.1,.2,.3}),q=vec({1,0,0,0});
    py::object dq=vec({std::cos(.05)/1.0,std::sin(.05)/std::sqrt(5.0),
                       2*std::sin(.05)/std::sqrt(5.0),0});
    finite_matrix(sensor.attr("jacobian_relative_quaternion")(a,g,q,dq));
    finite_matrix(sensor.attr("jacobian_relative_mrp")(a,g,q,vec({.01,-.02,.03})));
    finite_matrix(sensor.attr("jacobian_relative_angle_vector")(a,g,q,vec({.01,-.02,.03})));
}
void test_saturation_and_availability(py::object type) {
    py::object sensor=make_sensor(type,.5); const double expected[]={.5,.5,.5};
    close_vec(sensor.attr("noiseless_biasfree_measurement")(vec({1,1,1}),vec({0,0,0}),vec({1,0,0,0})),expected);
    CHECK(sensor.attr("availability")().cast<bool>()); sensor.attr("enabled")=false; CHECK(!sensor.attr("availability")().cast<bool>());
}
}
int main(int argc,char** argv) {
    py::scoped_interpreter interpreter{}; CHECK(argc==2);
    py::object type=load_adcs_subsystem_module(argv[1]).attr("Accelerometer");
    test_specific_force_and_mounting(type); test_bias_noise_and_jacobians(type);
    test_nonzero_attitude_jacobians_are_finite(type); test_saturation_and_availability(type);
}
