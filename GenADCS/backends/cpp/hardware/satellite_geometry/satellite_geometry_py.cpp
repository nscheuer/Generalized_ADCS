#include <pybind11/pybind11.h>
namespace py = pybind11;
namespace
{
struct BoundGeometry
{
    double mass;
    py::object inertia;
    py::list panels;
    py::object center_of_mass;
    BoundGeometry(double m, py::object i, py::object p, py::object c)
        : mass(m), inertia(std::move(i)), panels(p.is_none() ? py::list() : py::cast<py::list>(p)),
          center_of_mass(c.is_none() ? py::object(py::make_tuple(0.0, 0.0, 0.0)) : std::move(c))
    {
    }
};
bool equal(py::handle a, py::handle b)
{
    return py::module_::import("numpy").attr("array_equal")(a, b).cast<bool>();
}
} // namespace
void bind_satellite_geometry(py::module_ &module)
{
    py::class_<BoundGeometry>(module, "SatelliteGeometry")
        .def(py::init<double, py::object, py::object, py::object>(), py::arg("mass"), py::arg("inertia"),
             py::arg("panels") = py::none(), py::arg("center_of_mass") = py::none())
        .def_readwrite("mass", &BoundGeometry::mass)
        .def_readwrite("inertia", &BoundGeometry::inertia)
        .def_property(
            "panels", [](BoundGeometry &g) { return g.panels; },
            [](BoundGeometry &g, py::list v) { g.panels = std::move(v); })
        .def_readwrite("center_of_mass", &BoundGeometry::center_of_mass)
        .def("__eq__", [](const BoundGeometry &a, const BoundGeometry &b) {
            return a.mass == b.mass && equal(a.inertia, b.inertia) && equal(a.center_of_mass, b.center_of_mass) &&
                   equal(a.panels, b.panels);
        });
}
