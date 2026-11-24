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

void bind_native_models(py::module_ &module)
{
    module.def("magnetic_field_inertial_tesla", &env::magnetic_field_with_coefficients, py::arg("position_inertial_m"),
               py::arg("epoch_jd"), py::arg("g"), py::arg("h"), py::arg("nmax"));
    module.def("_schmidt_table", &env::schmidt_table);
    module.def(
        "read_shc",
        [](const std::filesystem::path &path, int nmin, int nmax) {
            auto [epochs, g, h] = env::load_shc(path, nmin, nmax);
            const py::ssize_t count = static_cast<py::ssize_t>(epochs.size()),
                              width = static_cast<py::ssize_t>(nmax + 1);
            py::array_t<double> ga({count, width, width}), ha({count, width, width});
            auto *gp = static_cast<double *>(ga.request().ptr);
            auto *hp = static_cast<double *>(ha.request().ptr);
            for (py::ssize_t t = 0; t < count; ++t)
                for (py::ssize_t n = 0; n < width; ++n)
                    for (py::ssize_t m = 0; m < width; ++m)
                    {
                        const std::size_t index = static_cast<std::size_t>((t * width + n) * width + m);
                        gp[index] = g[static_cast<std::size_t>(t)](n, m);
                        hp[index] = h[static_cast<std::size_t>(t)](n, m);
                    }
            return py::make_tuple(epochs, ga, ha);
        },
        py::arg("path"), py::arg("nmin"), py::arg("nmax"));
}
