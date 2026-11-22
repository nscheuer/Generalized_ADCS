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
struct NativeNrlmsise {};
std::array<double,7> history_from_python(const py::object& value) {
    std::array<double,7> result{};
    result.fill(-1.0);
    if (!value.is_none()) {
        const auto sequence=py::cast<py::sequence>(value);
        if (sequence.size()!=7) throw py::value_error("ap_array must contain seven Ap values");
        for (py::ssize_t i=0;i<7;++i) result[static_cast<std::size_t>(i)]=py::cast<double>(sequence[i]);
    }
    return result;
}
std::string body_name(const py::object& body) {
    if (py::isinstance<py::str>(body)) return py::cast<std::string>(body);
    const int id=py::cast<int>(body);
    if(id==0)return "sun"; if(id==1)return "earth"; if(id==2)return "moon";
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
    return std::move(output);
}
}

PYBIND11_MODULE(_environment, module) {
    module.doc() = "Native C/C++ environmental models with pybind11 bindings";
    py::class_<env::PlanetaryBody>(module, "PlanetaryBody")
        .def(py::init<std::string,double,double>(),py::arg("name"),py::arg("gravitational_parameter"),py::arg("reference_radius"))
        .def_readwrite("name",&env::PlanetaryBody::name)
        .def_readwrite("gravitational_parameter",&env::PlanetaryBody::gravitational_parameter)
        .def_readwrite("reference_radius",&env::PlanetaryBody::reference_radius)
        .def("__eq__",[](const env::PlanetaryBody& a,const env::PlanetaryBody& b){return a.name==b.name&&a.gravitational_parameter==b.gravitational_parameter&&a.reference_radius==b.reference_radius;});
    py::class_<env::Earth,env::PlanetaryBody>(module,"Earth").def(py::init<>());
    py::class_<env::JGravityModel>(module,"JGravityModel")
        .def(py::init<std::vector<int>,const Eigen::VectorXd&>(),py::arg("degrees")=std::vector<int>{2,3,4,5,6},py::arg("coefficients")=Eigen::VectorXd())
        .def("evaluate",&env::JGravityModel::evaluate,py::arg("position_inertial_m"),py::arg("epoch_jd")=0.0)
        .def("acceleration",&env::JGravityModel::acceleration,py::arg("position_inertial_m"),py::arg("epoch_jd")=0.0)
        .def_property_readonly("degrees",[](const env::JGravityModel& self){py::tuple t(self.degrees().size());for(std::size_t i=0;i<self.degrees().size();++i)t[i]=self.degrees()[i];return t;})
        .def_property_readonly("zonal_coefficients",&env::JGravityModel::zonal_coefficients)
        .def_property_readonly("degree",&env::JGravityModel::degree)
        .def_static("acceleration_inertial_m_s2",&env::gravity_acceleration_inertial_m_s2);
#define BIND_J_MODEL(TYPE) py::class_<env::TYPE,env::JGravityModel>(module,#TYPE).def(py::init<>())
    BIND_J_MODEL(J2);BIND_J_MODEL(J3);BIND_J_MODEL(J4);BIND_J_MODEL(J5);BIND_J_MODEL(J6);
#undef BIND_J_MODEL
    module.attr("JGravityModel").attr("degree")=py::none();module.attr("J2").attr("degree")=2;module.attr("J3").attr("degree")=3;module.attr("J4").attr("degree")=4;module.attr("J5").attr("degree")=5;module.attr("J6").attr("degree")=6;
    module.attr("EARTH_ZONAL_COEFFICIENTS")=py::module_::import("numpy").attr("array")(py::make_tuple(1.08262668e-3,-2.5324105e-6,-1.6198976e-6,-2.2729608e-7,5.406812e-7));module.attr("EARTH_MU_M3_S2")=3.986004418e14;module.attr("EARTH_REFERENCE_RADIUS_M")=6378137.0;
    module.def("gravity_acceleration_inertial_m_s2",&env::gravity_acceleration_inertial_m_s2);
    py::class_<env::Environment>(module,"Environment").def(py::init([](py::object body,py::object atmosphere,py::object magnetic,py::object gravity,py::object ephemeris){if(body.is_none())body=py::cast(env::Earth());return env::Environment(std::move(body),std::move(atmosphere),std::move(magnetic),std::move(gravity),std::move(ephemeris));}),py::arg("planetary_body")=py::none(),py::arg("atmosphere")=py::none(),py::arg("magnetic_field")=py::none(),py::arg("j_gravity")=py::none(),py::arg("ephemeris")=py::none()).def_readwrite("planetary_body",&env::Environment::planetary_body).def_readwrite("atmosphere",&env::Environment::atmosphere).def_readwrite("magnetic_field",&env::Environment::magnetic_field).def_readwrite("j_gravity",&env::Environment::j_gravity).def_readwrite("ephemeris",&env::Environment::ephemeris);
    py::class_<env::AtmosphericModel>(module,"AtmosphericModel").def_static("density",[](const Eigen::Vector3d& p,double jd){return env::atmospheric_density(p,jd);}).def_static("velocity",&env::co_rotating_atmosphere_velocity,py::arg("position_inertial_m"),py::arg("epoch_jd"),py::arg("angular_velocity_inertial_rad_s")=Eigen::Vector3d(0.0,0.0,7.2921150e-5));
    module.def("atmospheric_density",&env::atmospheric_density);module.def("co_rotating_atmosphere_velocity",&env::co_rotating_atmosphere_velocity,py::arg("position_inertial_m"),py::arg("epoch_jd"),py::arg("angular_velocity_inertial_rad_s")=Eigen::Vector3d(0.0,0.0,7.2921150e-5));
    auto density_geodetic=[](int doy,double sec,double alt,double lat,double lon,double f107,double f107a,double ap,bool effective,py::object history){return env::nrlmsise_density_geodetic(doy,sec,alt,lat,lon,f107,f107a,ap,effective,history_from_python(history));};
    auto density_inertial=[](const Eigen::Vector3d& p,double jd,double f107,double f107a,double ap,bool effective,py::object history){return env::nrlmsise_density(p,jd,f107,f107a,ap,effective,history_from_python(history));};
    module.def("nrlmsise00_density_geodetic",density_geodetic,py::arg("day_of_year_utc"),py::arg("seconds_utc"),py::arg("altitude_km"),py::arg("latitude_deg"),py::arg("longitude_deg"),py::arg("f107")=150.0,py::arg("f107a")=150.0,py::arg("ap")=4.0,py::arg("effective_density")=true,py::arg("ap_array")=py::none());
    module.def("nrlmsise00_density",density_inertial,py::arg("position_inertial_m"),py::arg("epoch_jd"),py::arg("f107")=150.0,py::arg("f107a")=150.0,py::arg("ap")=4.0,py::arg("effective_density")=true,py::arg("ap_array")=py::none());
    py::class_<NativeNrlmsise>(module,"NRLMSISE00").def(py::init<>()).def_static("density",density_inertial,py::arg("position_inertial_m"),py::arg("epoch_jd"),py::arg("f107")=150.0,py::arg("f107a")=150.0,py::arg("ap")=4.0,py::arg("effective_density")=true,py::arg("ap_array")=py::none()).def_static("density_geodetic",density_geodetic,py::arg("day_of_year_utc"),py::arg("seconds_utc"),py::arg("altitude_km"),py::arg("latitude_deg"),py::arg("longitude_deg"),py::arg("f107")=150.0,py::arg("f107a")=150.0,py::arg("ap")=4.0,py::arg("effective_density")=true,py::arg("ap_array")=py::none());
    module.attr("NRMLSISE")=module.attr("NRLMSISE00");
    py::class_<env::DE421>(module,"DE421").def(py::init([](py::object filepath){return filepath.is_none()?env::DE421():env::DE421(py::cast<std::filesystem::path>(filepath));}),py::arg("filepath")=py::none()).def_property_readonly("available_bodies",&env::DE421::available_bodies).def_property_readonly("filepath",&env::DE421::filepath)
      .def("position_eci",[](const env::DE421& self,py::object target,py::object epoch,py::object observer){const std::string t=body_name(target),o=body_name(observer);return vectorized_epoch(epoch,[&](double jd){return self.position_eci(t,jd,o);});},py::arg("target"),py::arg("epoch_jd_tdb"),py::arg("observer")="earth")
      .def("velocity_eci",[](const env::DE421& self,py::object target,py::object epoch,py::object observer){const std::string t=body_name(target),o=body_name(observer);return vectorized_epoch(epoch,[&](double jd){return self.velocity_eci(t,jd,o);});},py::arg("target"),py::arg("epoch_jd_tdb"),py::arg("observer")="earth")
      .def("state_eci",[](const env::DE421& self,py::object target,double jd,py::object observer){return self.state_eci(body_name(target),jd,body_name(observer));},py::arg("target"),py::arg("epoch_jd_tdb"),py::arg("observer")="earth")
      .def("position",&env::DE421::position).def("velocity",&env::DE421::velocity).def("sun_position_eci",[](const env::DE421& self,double jd){return self.position(0,jd);});
    module.attr("DE421").attr("DATA_FILENAME")="de421.bsp";module.attr("BODY_SUN")=0;module.attr("BODY_EARTH")=1;module.attr("BODY_MOON")=2;
    module.def("eclipse_state",&env::eclipse_state);
    module.def("magnetic_field_inertial_tesla",&env::magnetic_field_with_coefficients,py::arg("position_inertial_m"),py::arg("epoch_jd"),py::arg("g"),py::arg("h"),py::arg("nmax"));
    module.def("_schmidt_table",&env::schmidt_table);
    module.def("read_shc",[](const std::filesystem::path& path,int nmin,int nmax){auto [epochs,g,h]=env::load_shc(path,nmin,nmax);const py::ssize_t count=static_cast<py::ssize_t>(epochs.size()), width=static_cast<py::ssize_t>(nmax+1);py::array_t<double> ga({count,width,width}),ha({count,width,width});auto* gp=static_cast<double*>(ga.request().ptr);auto* hp=static_cast<double*>(ha.request().ptr);for(py::ssize_t t=0;t<count;++t)for(py::ssize_t n=0;n<width;++n)for(py::ssize_t m=0;m<width;++m){const std::size_t index=static_cast<std::size_t>((t*width+n)*width+m);gp[index]=g[static_cast<std::size_t>(t)](n,m);hp[index]=h[static_cast<std::size_t>(t)](n,m);}return py::make_tuple(epochs,ga,ha);},py::arg("path"),py::arg("nmin"),py::arg("nmax"));
    module.def("ephemeris_position",[](int id,double jd){static const env::DE421 kernel;return kernel.position(id,jd);});
    module.def("ephemeris_velocity",[](int id,double jd){static const env::DE421 kernel;return kernel.velocity(id,jd);});
    py::class_<env::MagneticFieldModel>(module,"MagneticFieldModel").def("field",&env::MagneticFieldModel::field).def("field_inertial_tesla",&env::MagneticFieldModel::field_inertial_tesla);
    py::class_<env::IGRF14,env::MagneticFieldModel>(module,"IGRF14").def(py::init<>()).def("field",&env::IGRF14::field).def("field_inertial_tesla",&env::IGRF14::field_inertial_tesla);
    py::class_<env::CHAOS8,env::MagneticFieldModel>(module,"CHAOS8").def(py::init<bool>(),py::arg("include_static")=true).def_property("include_static",&env::CHAOS8::include_static,[](env::CHAOS8&,bool){throw py::attribute_error("include_static is read-only after construction");}).def("field",&env::CHAOS8::field).def("field_inertial_tesla",&env::CHAOS8::field_inertial_tesla);
    py::class_<env::WMM,env::MagneticFieldModel>(module,"WMM").def(py::init<>()).def("field",[](const env::WMM& self,const Eigen::Vector3d& p,double jd){try{return self.field(p,jd);}catch(const std::runtime_error& e){PyErr_SetString(PyExc_NotImplementedError,e.what());throw py::error_already_set();}}).def("field_inertial_tesla",[](const env::WMM& self,const Eigen::Vector3d& p,double jd){try{return self.field_inertial_tesla(p,jd);}catch(const std::runtime_error& e){PyErr_SetString(PyExc_NotImplementedError,e.what());throw py::error_already_set();}});
}
