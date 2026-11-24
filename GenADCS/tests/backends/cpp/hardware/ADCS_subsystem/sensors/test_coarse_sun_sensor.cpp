#include "../check.hpp"
#include <pybind11/numpy.h>
#include <cmath>
#include <initializer_list>
namespace {
py::array_t<double> vec(std::initializer_list<double> x){py::array_t<double>a(static_cast<py::ssize_t>(x.size()));auto o=a.mutable_unchecked<1>();py::ssize_t i=0;for(double v:x)o(i++)=v;return a;}
py::object sensor(py::object type,py::object normal=py::none(),double sensitivity=2,double limit=10,double variance=.01,double bias=0,double fov=1.0471975511965976,py::object orientation=py::none(),py::object mis=py::none()){
    if (normal.is_none()) normal=vec({1,0,0});
    if (orientation.is_none()) orientation=vec({1,0,0,0});
    if (mis.is_none()) mis=vec({1,0,0,0});
    return type(vec({0,0,0}),orientation,mis,normal,sensitivity,limit,variance,bias,fov);
}
void test_cosine_projection_and_frontside(py::object type){
    auto s=sensor(type);py::object q=vec({1,0,0,0});
    CHECK(std::abs(s.attr("noiseless_biasfree_measurement")(vec({1,0,0}),q).cast<double>()-2.)<1e-12);
    auto back=sensor(type,py::none(),2,10,.01,-.5);
    CHECK(back.attr("noiseless_biasfree_measurement")(vec({-1,0,0}),q).cast<double>()==0.);
    CHECK(back.attr("noiseless_biased_measurement")(vec({-1,0,0}),q).cast<double>()==0.);
    auto rotated=sensor(type,vec({0,1,0}),2,10,.01,0,1.0471975511965976,
                        vec({std::sqrt(.5),0,0,std::sqrt(.5)}),
                        vec({std::cos(.1),std::sin(.1),0,0}));
    CHECK(std::abs(rotated.attr("noiseless_biasfree_measurement")(vec({1,0,0}),q).cast<double>()-
                   2*std::cos(.2))<1e-12);
}
void test_bias_saturation_and_noise(py::object type){
    auto s=sensor(type,py::none(),3,2,.00000001,.25);
    py::object q=vec({1,0,0,0});
    CHECK(s.attr("noiseless_biasfree_measurement")(vec({1,0,0}),q).cast<double>()==2.);
    CHECK(s.attr("noiseless_biased_measurement")(vec({1,0,0}),q).cast<double>()==2.);
    CHECK(s.attr("jacobian_bias")().cast<double>()==1.);
    auto noisy=sensor(type,py::none(),2,3,1e-8);
    for(const char* name:{"noisy_biasfree_measurement","noisy_biased_measurement"}){
        double x=noisy.attr(name)(vec({.5,0,std::sqrt(.75)}),q).cast<double>();CHECK(std::isfinite(x)&&x>=0&&x<=3);
    }
}
void test_jacobians_and_availability_information(py::object type){
    auto s=sensor(type,vec({0,1,0}));py::object q=vec({1,0,0,0}),sun=vec({0,0,1});
    auto jq=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(s.attr("jacobian_relative_quaternion")(sun,q,q));
    auto jm=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(s.attr("jacobian_relative_mrp")(sun,q,vec({0,0,0})));
    auto ja=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(s.attr("jacobian_relative_angle_vector")(sun,q,vec({0,0,0})));
    for(auto a:{jq,jm,ja}){CHECK(a&&a.ndim()==1&&a.shape(0)==3);auto v=a.unchecked<1>();for(int i=0;i<3;++i)CHECK(std::isfinite(v(i)));}
    const double expected_q[]={-4,0,0};
    auto qv=jq.unchecked<1>();auto mv=jm.unchecked<1>();auto av=ja.unchecked<1>();
    for(int i=0;i<3;++i){CHECK(std::abs(qv(i)-expected_q[i])<1e-12);CHECK(std::abs(mv(i)-2*expected_q[i])<1e-12);CHECK(std::abs(av(i)-.5*expected_q[i])<1e-12);}
    auto delta_q=vec({std::cos(.05),std::sin(.05),0,0});
    for(auto jac:{s.attr("jacobian_relative_quaternion")(vec({.2,-.1,.97}),q,delta_q),
                  s.attr("jacobian_relative_mrp")(vec({.2,-.1,.97}),q,vec({.02,-.03,.01})),
                  s.attr("jacobian_relative_angle_vector")(vec({.2,-.1,.97}),q,vec({.02,-.03,.01}))}){
        auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(jac);CHECK(a&&a.ndim()==1&&a.shape(0)==3);auto v=a.unchecked<1>();for(int i=0;i<3;++i)CHECK(std::isfinite(v(i)));
    }
    auto f=sensor(type,vec({0,0,1}),2,10,.5,0,.3);
    py::object aligned=vec({0,0,1});CHECK(f.attr("availability")(aligned,q).cast<bool>());
    CHECK(!f.attr("availability")(aligned,q,false).cast<bool>());
    CHECK(!f.attr("availability")(vec({0,0,-1}),q).cast<bool>());
    CHECK(!f.attr("availability")(vec({std::sin(.5),0,std::cos(.5)}),q).cast<bool>());
    auto info=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(f.attr("fisher_information_matrix")(vec({0,std::sin(.2),std::cos(.2)}),q));
    CHECK(info&&info.ndim()==2&&info.shape(0)==3&&info.shape(1)==3);
    auto iv=info.unchecked<2>();const double expected_information=8*std::sin(.2)*std::sin(.2);
    CHECK(std::abs(iv(0,0)-expected_information)<1e-12);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){CHECK(std::isfinite(iv(i,j)));if(i!=0||j!=0)CHECK(std::abs(iv(i,j))<1e-12);}
    auto zero=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(f.attr("fisher_information_matrix")(aligned,q,false));
    auto zv=zero.unchecked<2>();for(int i=0;i<3;++i)for(int j=0;j<3;++j)CHECK(zv(i,j)==0.);
}
}
int main(int argc,char**argv){py::scoped_interpreter interpreter{};CHECK(argc==2);auto type=load_adcs_subsystem_module(argv[1]).attr("CoarseSunSensor");test_cosine_projection_and_frontside(type);test_bias_saturation_and_noise(type);test_jacobians_and_availability_information(type);}
