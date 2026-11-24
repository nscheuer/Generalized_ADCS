#include <pybind11/pybind11.h>
namespace py = pybind11;
void bind_planetary_body_planetary_body(py::module_ &module);
void bind_planetary_body_earth(py::module_ &module);
void bind_gravity_j_gravity_model(py::module_ &module);
void bind_environment(py::module_ &module);
void bind_atmosphere_atmospheric_model(py::module_ &module);
void bind_atmosphere_NRMLSISE00_nrlmsise00(py::module_ &module);
void bind_ephemeris_de421_de421(py::module_ &module);
void bind_ephemeris_ephemeris_model(py::module_ &module);
void bind_native_models(py::module_ &module);
void bind_magnetic_field_model_magnetic_field_model(py::module_ &module);
void bind_magnetic_field_model_igrf14_igrf14(py::module_ &module);
void bind_magnetic_field_model_chaos8_chaos8(py::module_ &module);
void bind_magnetic_field_model_wmm_wmm(py::module_ &module);
void bind_gravity_j2(py::module_ &module);
void bind_gravity_j3(py::module_ &module);
void bind_gravity_j4(py::module_ &module);
void bind_gravity_j5(py::module_ &module);
void bind_gravity_j6(py::module_ &module);

PYBIND11_MODULE(_environment, module)
{
    module.doc() = "Native C/C++ environmental models with pybind11 bindings";
    bind_planetary_body_planetary_body(module);
    bind_planetary_body_earth(module);
    bind_gravity_j_gravity_model(module);
    bind_environment(module);
    bind_atmosphere_atmospheric_model(module);
    bind_atmosphere_NRMLSISE00_nrlmsise00(module);
    bind_ephemeris_de421_de421(module);
    bind_ephemeris_ephemeris_model(module);
    bind_native_models(module);
    bind_magnetic_field_model_magnetic_field_model(module);
    bind_magnetic_field_model_igrf14_igrf14(module);
    bind_magnetic_field_model_chaos8_chaos8(module);
    bind_magnetic_field_model_wmm_wmm(module);
    bind_gravity_j2(module);
    bind_gravity_j3(module);
    bind_gravity_j4(module);
    bind_gravity_j5(module);
    bind_gravity_j6(module);
}
