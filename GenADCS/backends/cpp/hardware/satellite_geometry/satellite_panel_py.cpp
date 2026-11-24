#include <pybind11/pybind11.h>
namespace py = pybind11;
namespace
{
struct BoundPanel
{
    py::object position, orientation, dimensions;
    double eta_s, eta_d, eta_a, cd;
    BoundPanel(py::object p, py::object o, py::object d, double s, double diffuse, double a, double drag)
        : position(std::move(p)), orientation(std::move(o)), dimensions(std::move(d)), eta_s(s), eta_d(diffuse),
          eta_a(a), cd(drag)
    {
    }
};
bool equal(py::handle a, py::handle b)
{
    return py::module_::import("numpy").attr("array_equal")(a, b).cast<bool>();
}
} // namespace
void bind_satellite_panel(py::module_ &module)
{
    py::class_<BoundPanel>(module, "SatellitePanel")
        .def(py::init<py::object, py::object, py::object, double, double, double, double>(), py::arg("position"),
             py::arg("orientation"), py::arg("dimensions"), py::arg("eta_s") = 0.0, py::arg("eta_d") = 0.0,
             py::arg("eta_a") = 1.0, py::arg("cd") = 2.2)
        .def_readwrite("position", &BoundPanel::position)
        .def_readwrite("orientation", &BoundPanel::orientation)
        .def_readwrite("dimensions", &BoundPanel::dimensions)
        .def_readwrite("eta_s", &BoundPanel::eta_s)
        .def_readwrite("eta_d", &BoundPanel::eta_d)
        .def_readwrite("eta_a", &BoundPanel::eta_a)
        .def_readwrite("cd", &BoundPanel::cd)
        .def("__eq__", [](const BoundPanel &a, const BoundPanel &b) {
            return equal(a.position, b.position) && equal(a.orientation, b.orientation) &&
                   equal(a.dimensions, b.dimensions) && a.eta_s == b.eta_s && a.eta_d == b.eta_d &&
                   a.eta_a == b.eta_a && a.cd == b.cd;
        });
}
