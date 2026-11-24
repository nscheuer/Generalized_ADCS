#include "environment.hpp"
#include "gravity/j_gravity_model.hpp"
#include "planetary_body/planetary_body.hpp"
#include "atmosphere/atmospheric_model.hpp"
#include "native_models.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <array>
#include <string>
#include <utility>

namespace py = pybind11;
namespace env = omni::environment;

namespace {
std::string body_name(const py::object& body) {
    if (py::isinstance<py::str>(body)) return py::cast<std::string>(body);
    const int id=py::cast<int>(body);
    if (id == 0) return "sun";
    if (id == 1) return "earth";
    if (id == 2) return "moon";
    switch(id) { case 199:return "mercury";case 299:return "venus";case 3:return "earth-moon barycenter";case 499:return "mars";case 599:return "jupiter";case 699:return "saturn";case 799:return "uranus";case 899:return "neptune";case 999:return "pluto";case 301:return "moon";case 10:return "sun";case 399:return "earth";default:throw py::value_error("unsupported DE421 body identifier"); }
}
template<class Function> py::object vectorized_epoch(const py::object& epoch, Function function) {
    py::array_t<double,py::array::c_style|py::array::forcecast> values(epoch);
    const auto info=values.request();
    if(info.ndim==0) return py::cast(function(*static_cast<const double*>(info.ptr)));
    std::vector<py::ssize_t> shape(info.shape.begin(),info.shape.end());
    shape.push_back(3);
    py::array_t<double> output(shape);
    const auto* input=static_cast<const double*>(info.ptr);
    auto* result=static_cast<double*>(output.request().ptr);
    const std::size_t count=static_cast<std::size_t>(info.size);
    for(std::size_t i=0;i<count;++i) { const Eigen::Vector3d value=function(input[i]); for(int axis=0;axis<3;++axis) result[3U*i+static_cast<std::size_t>(axis)]=value[axis]; }
    return output;
}
}
void bind_ephemeris_de421_de421(py::module_& module) {
    py::class_<env::DE421, std::shared_ptr<env::DE421>>(module,"DE421").def(py::init([](py::object filepath){return filepath.is_none()?env::DE421():env::DE421(py::cast<std::filesystem::path>(filepath));}),py::arg("filepath")=py::none()).def_property_readonly("available_bodies",&env::DE421::available_bodies).def_property_readonly("filepath",&env::DE421::filepath)
      .def("position_eci",[](const env::DE421& self,py::object target,py::object epoch,py::object observer){const std::string t=body_name(target),o=body_name(observer);return vectorized_epoch(epoch,[&](double jd){return self.position_eci(t,jd,o);});},py::arg("target"),py::arg("epoch_jd_tdb"),py::arg("observer")="earth")
      .def("velocity_eci",[](const env::DE421& self,py::object target,py::object epoch,py::object observer){const std::string t=body_name(target),o=body_name(observer);return vectorized_epoch(epoch,[&](double jd){return self.velocity_eci(t,jd,o);});},py::arg("target"),py::arg("epoch_jd_tdb"),py::arg("observer")="earth")
      .def("state_eci",[](const env::DE421& self,py::object target,double jd,py::object observer){return self.state_eci(body_name(target),jd,body_name(observer));},py::arg("target"),py::arg("epoch_jd_tdb"),py::arg("observer")="earth")
      .def("position",&env::DE421::position).def("velocity",&env::DE421::velocity).def("sun_position_eci",[](const env::DE421& self,double jd){return self.position(0,jd);});
    module.attr("DE421").attr("DATA_FILENAME")="de421.bsp";module.attr("BODY_SUN")=0;module.attr("BODY_EARTH")=1;module.attr("BODY_MOON")=2;
    module.def("ephemeris_position",[](int id,double jd){static const env::DE421 kernel;return kernel.position(id,jd);});
    module.def("ephemeris_velocity",[](int id,double jd){static const env::DE421 kernel;return kernel.velocity(id,jd);});
}
