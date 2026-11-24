#include "atmosphere/atmospheric_model.hpp"
#include "environment.hpp"
#include "gravity/j_gravity_model.hpp"
#include "native_models.hpp"
#include "planetary_body/planetary_body.hpp"

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

namespace
{
struct NativeNrlmsise : env::AtmosphericModel
{
    double density(const Eigen::Vector3d &p, double jd) const override
    {
        std::array<double, 7> history{};
        history.fill(-1.0);
        return env::nrlmsise_density(p, jd, 150.0, 150.0, 4.0, true, history);
    }
};
std::array<double, 7> history_from_python(const py::object &value)
{
    std::array<double, 7> result{};
    result.fill(-1.0);
    if (!value.is_none())
    {
        const auto sequence = py::cast<py::sequence>(value);
        if (sequence.size() != 7)
            throw py::value_error("ap_array must contain seven Ap values");
        for (py::ssize_t i = 0; i < 7; ++i)
            result[static_cast<std::size_t>(i)] = py::cast<double>(sequence[i]);
    }
    return result;
}

} // namespace
void bind_atmosphere_NRMLSISE00_nrlmsise00(py::module_ &module)
{
    auto density_geodetic = [](int doy, double sec, double alt, double lat, double lon, double f107, double f107a,
                               double ap, bool effective, py::object history) {
        return env::nrlmsise_density_geodetic(doy, sec, alt, lat, lon, f107, f107a, ap, effective,
                                              history_from_python(history));
    };
    auto density_inertial = [](const Eigen::Vector3d &p, double jd, double f107, double f107a, double ap,
                               bool effective, py::object history) {
        return env::nrlmsise_density(p, jd, f107, f107a, ap, effective, history_from_python(history));
    };
    module.def("nrlmsise00_density_geodetic", density_geodetic, py::arg("day_of_year_utc"), py::arg("seconds_utc"),
               py::arg("altitude_km"), py::arg("latitude_deg"), py::arg("longitude_deg"), py::arg("f107") = 150.0,
               py::arg("f107a") = 150.0, py::arg("ap") = 4.0, py::arg("effective_density") = true,
               py::arg("ap_array") = py::none());
    module.def("nrlmsise00_density", density_inertial, py::arg("position_inertial_m"), py::arg("epoch_jd"),
               py::arg("f107") = 150.0, py::arg("f107a") = 150.0, py::arg("ap") = 4.0,
               py::arg("effective_density") = true, py::arg("ap_array") = py::none());
    py::class_<NativeNrlmsise, env::AtmosphericModel, std::shared_ptr<NativeNrlmsise>>(module, "NRLMSISE00")
        .def(py::init<>())
        .def_static("density", density_inertial, py::arg("position_inertial_m"), py::arg("epoch_jd"),
                    py::arg("f107") = 150.0, py::arg("f107a") = 150.0, py::arg("ap") = 4.0,
                    py::arg("effective_density") = true, py::arg("ap_array") = py::none())
        .def_static("density_geodetic", density_geodetic, py::arg("day_of_year_utc"), py::arg("seconds_utc"),
                    py::arg("altitude_km"), py::arg("latitude_deg"), py::arg("longitude_deg"), py::arg("f107") = 150.0,
                    py::arg("f107a") = 150.0, py::arg("ap") = 4.0, py::arg("effective_density") = true,
                    py::arg("ap_array") = py::none());
    module.attr("NRMLSISE") = module.attr("NRLMSISE00");
}
