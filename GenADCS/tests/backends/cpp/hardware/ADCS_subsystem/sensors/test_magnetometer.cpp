#include "../check.hpp"
#include <pybind11/numpy.h>

#include <cmath>
#include <initializer_list>

namespace {
py::array_t<double> vector(std::initializer_list<double> values) {
    py::array_t<double> result(static_cast<py::ssize_t>(values.size()));
    auto out=result.mutable_unchecked<1>(); py::ssize_t i=0;
    for(double value:values) out(i++)=value;
    return result;
}
py::array_t<double> diagonal(double x,double y,double z) {
    py::array_t<double> result({3,3}); auto out=result.mutable_unchecked<2>();
    for(py::ssize_t i=0;i<3;++i) for(py::ssize_t j=0;j<3;++j) out(i,j)=0.0;
    out(0,0)=x; out(1,1)=y; out(2,2)=z; return result;
}
py::object make_sensor(const py::object& type,double limit=1.0,py::object noise=py::none(),
                       py::object bias=py::none(),py::object orientation=py::none(),
                       py::object misalignment=py::none()) {
    if(noise.is_none()) noise=diagonal(1e-12,1e-12,1e-12);
    if(bias.is_none()) bias=vector({0,0,0});
    if(orientation.is_none()) orientation=vector({1,0,0,0});
    if(misalignment.is_none()) misalignment=vector({1,0,0,0});
    return type(vector({0,0,0}),orientation,misalignment,limit,noise,bias);
}
void check_vector(py::handle actual,const double* expected,py::ssize_t n,double tolerance=1e-12) {
    auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(actual);
    CHECK(a && a.ndim()==1 && a.shape(0)==n); auto v=a.unchecked<1>();
    for(py::ssize_t i=0;i<n;++i) CHECK(std::abs(v(i)-expected[i])<=tolerance);
}
void check_matrix(py::handle actual,const double expected[3][3],double tolerance=1e-12) {
    auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(actual);
    CHECK(a && a.ndim()==2 && a.shape(0)==3 && a.shape(1)==3); auto v=a.unchecked<2>();
    for(py::ssize_t i=0;i<3;++i) for(py::ssize_t j=0;j<3;++j) CHECK(std::abs(v(i,j)-expected[i][j])<=tolerance);
}
void test_noiseless_measurements_rotate_inertial_field_to_sensor_frame(const py::object& type) {
    const double h=std::sqrt(0.5);
    py::object q_att=vector({h,0,0,h});
    py::object q_mount=vector({h,h,0,0});
    py::object q_mis=vector({h,0,h,0});
    py::object sensor=make_sensor(type,1.0,py::none(),py::none(),q_mount,q_mis);
    const double field[]={2e-5,-3e-5,5e-5};
    py::object f=vector({field[0],field[1],field[2]});
    const double expected[]={2e-5,-5e-5,-3e-5};
    check_vector(sensor.attr("noiseless_biasfree_measurement")(f,q_att),expected,3);
    check_vector(sensor.attr("noiseless_biased_measurement")(f,q_att),expected,3);
}
void test_bias_is_additive_before_componentwise_saturation(const py::object& type) {
    py::object sensor=make_sensor(type,4e-5,py::none(),vector({1e-5,-2e-5,0}));
    py::object field=vector({4e-5,-4e-5,2e-5}), q=vector({1,0,0,0});
    const double expected[]={4e-5,-4e-5,2e-5};
    check_vector(sensor.attr("noiseless_biased_measurement")(field,q),expected,3,1e-15);
    check_vector(sensor.attr("noiseless_biasfree_measurement")(field,q),expected,3,1e-15);
    const double eye[3][3]={{1,0,0},{0,1,0},{0,0,1}};
    check_matrix(sensor.attr("jacobian_bias")(),eye);
}
void test_measurement_saturates_each_axis(const py::object& type) {
    py::object sensor=make_sensor(type,3e-5);
    const double input[]={5e-5,-4e-5,1e-5}, expected[]={3e-5,-3e-5,1e-5};
    check_vector(sensor.attr("noiseless_biasfree_measurement")(vector({input[0],input[1],input[2]}),vector({1,0,0,0})),expected,3,1e-15);
}
void test_noise_is_finite_and_measurement_shaped(const py::object& type,const char* method) {
    py::object sensor=make_sensor(type,1.0,diagonal(1e-10,4e-10,9e-10),vector({1e-6,2e-6,3e-6}));
    py::object result=sensor.attr(method)(vector({0,0,0}),vector({1,0,0,0}));
    auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(result);
    CHECK(a && a.ndim()==1 && a.shape(0)==3); auto v=a.unchecked<1>();
    for(py::ssize_t i=0;i<3;++i) CHECK(std::isfinite(v(i)));
}
void test_attitude_jacobians_at_identity(const py::object& type) {
    py::object sensor=make_sensor(type), field=vector({2e-5,-3e-5,4e-5}), q=vector({1,0,0,0});
    const double expected[3][3]={{0,8e-5,6e-5},{-8e-5,0,4e-5},{-6e-5,-4e-5,0}};
    check_matrix(sensor.attr("jacobian_relative_quaternion")(field,q,q),expected,1e-14);
    double twice[3][3]{}; double half[3][3]{};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) { twice[i][j]=2*expected[i][j]; half[i][j]=0.5*expected[i][j]; }
    check_matrix(sensor.attr("jacobian_relative_mrp")(field,q,vector({0,0,0})),twice,1e-14);
    check_matrix(sensor.attr("jacobian_relative_angle_vector")(field,q,vector({0,0,0})),half,1e-14);
}
void test_nonzero_relative_parameter_jacobians_are_finite(const py::object& type) {
    py::object sensor=make_sensor(type), field=vector({2e-5,1e-5,-3e-5}), q=vector({1,0,0,0});
    const double h=std::sin(0.1), delta_q[]={std::cos(0.1),h/std::sqrt(6.0),2*h/std::sqrt(6.0),-h/std::sqrt(6.0)};
    py::object jacobians[3]={sensor.attr("jacobian_relative_quaternion")(field,q,vector({delta_q[0],delta_q[1],delta_q[2],delta_q[3]})),
        sensor.attr("jacobian_relative_mrp")(field,q,vector({.02,-.01,.03})),
        sensor.attr("jacobian_relative_angle_vector")(field,q,vector({.02,-.01,.03}))};
    for(const auto& jac:jacobians) {
        auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(jac);
        CHECK(a && a.ndim()==2 && a.shape(0)==3 && a.shape(1)==3); auto v=a.unchecked<2>();
        for(py::ssize_t i=0;i<3;++i) for(py::ssize_t j=0;j<3;++j) CHECK(std::isfinite(v(i,j)));
    }
}
void test_availability_and_fisher_information(const py::object& type) {
    py::object sensor=make_sensor(type,1.0,diagonal(2,3,4));
    py::object field=vector({1,2,3}), q=vector({1,0,0,0}), covariance=diagonal(.1,.2,.3);
    CHECK(sensor.attr("availability")().cast<bool>());
    const double cross[3][3]={{0,-3,2},{3,0,-1},{-2,1,0}};
    double expected[3][3]{}; const double invdiag[]={1/2.1,1/3.2,1/4.3};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k) expected[i][j]+=cross[k][i]*invdiag[k]*cross[k][j];
    check_matrix(sensor.attr("fisher_information_matrix")(field,q,covariance),expected,1e-12);
    sensor.attr("enabled")=false;
    CHECK(!sensor.attr("availability")().cast<bool>());
    const double zero[3][3]{};
    check_matrix(sensor.attr("fisher_information_matrix")(field,q,covariance),zero);
}
}
int main(int argc,char** argv) {
    py::scoped_interpreter interpreter{}; CHECK(argc==2);
    py::object type=load_adcs_subsystem_module(argv[1]).attr("Magnetometer");
    test_noiseless_measurements_rotate_inertial_field_to_sensor_frame(type);
    test_bias_is_additive_before_componentwise_saturation(type);
    test_measurement_saturates_each_axis(type);
    test_noise_is_finite_and_measurement_shaped(type,"noisy_biasfree_measurement");
    test_noise_is_finite_and_measurement_shaped(type,"noisy_biased_measurement");
    test_attitude_jacobians_at_identity(type);
    test_nonzero_relative_parameter_jacobians_are_finite(type);
    test_availability_and_fisher_information(type);
}
