#include <pybind11/embed.h>
#include <pybind11/eval.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;

#define CHECK(condition) do { \
    if (!(condition)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #condition); \
} while (false)

py::module_ geometry_module(const char* extension_path) {
    py::object util = py::module_::import("importlib.util");
    py::object spec = util.attr("spec_from_file_location")("_satellite_geometry", extension_path);
    py::object module = util.attr("module_from_spec")(spec);
    spec.attr("loader").attr("exec_module")(module);
    return py::reinterpret_borrow<py::module_>(module);
}

py::object get(const py::object& instance, const char* name) {
    return instance.attr(name).cast<py::object>();
}

bool equal(py::handle left, py::handle right) {
    return PyObject_RichCompareBool(left.ptr(), right.ptr(), Py_EQ) == 1;
}

py::object make_panel(const py::module_& module) {
    return module.attr("SatellitePanel")(
        py::make_tuple(1.0, 2.0, 3.0),
        py::make_tuple(0.0, 0.0, 1.0),
        py::make_tuple(2.0, 1.0));
}

void test_geometry_defaults_mutation_and_independent_lists(const py::module_& module) {
    py::object geometry_type = module.attr("SatelliteGeometry");
    py::object first = geometry_type(1.0, py::module_::import("numpy").attr("eye")(3));
    py::object second = geometry_type(2.0, py::module_::import("numpy").attr("eye")(3));
    CHECK(get(first, "mass").cast<double>() == 1.0);
    CHECK(py::len(get(first, "panels")) == 0);
    CHECK(equal(get(first, "center_of_mass"), py::make_tuple(0.0, 0.0, 0.0)));
    CHECK(!get(first, "panels").is(get(second, "panels")));

    first.attr("mass") = 6.5;
    first.attr("center_of_mass") = py::make_tuple(0.25, -0.1, 0.0);
    CHECK(get(first, "mass").cast<double>() == 6.5);
    CHECK(equal(get(first, "center_of_mass"), py::make_tuple(0.25, -0.1, 0.0)));
    py::object panel = make_panel(module);
    get(first, "panels").attr("append")(panel);
    py::object stored_panel = get(first, "panels").attr("__getitem__")(0).cast<py::object>();
    stored_panel.attr("cd") = 2.6;
    CHECK(py::len(get(first, "panels")) == 1);
    CHECK(get(panel, "cd").cast<double>() == 2.6);
    CHECK(py::len(get(second, "panels")) == 0);
}

void test_geometry_retains_constructor_inputs_and_panel_list_alias(const py::module_& module) {
    py::dict scope;
    py::exec("class Retained: pass", scope);
    py::object retained_type = scope["Retained"];
    py::object inertia = retained_type();
    py::object center = retained_type();
    py::object inertia_ref = py::module_::import("weakref").attr("ref")(inertia);
    py::object center_ref = py::module_::import("weakref").attr("ref")(center);
    py::list panels;
    py::object panel = make_panel(module);
    panels.append(panel);
    py::object geometry = module.attr("SatelliteGeometry")(
        4.0, inertia, panels, py::arg("center_of_mass") = center);
    inertia = py::none();
    center = py::none();
    py::module_::import("gc").attr("collect")();
    py::object held_inertia = inertia_ref();
    py::object held_center = center_ref();
    CHECK(get(geometry, "inertia").is(held_inertia));
    CHECK(get(geometry, "center_of_mass").is(held_center));
    CHECK(get(geometry, "panels").is(panels));

    panels.append(make_panel(module));
    CHECK(py::len(get(geometry, "panels")) == 2);
}

void test_geometry_value_equality_repr_slots_and_argument_errors(const py::module_& module) {
    py::object panel_a = make_panel(module);
    py::object panel_b = make_panel(module);
    CHECK(panel_a.equal(panel_b));
    py::object numpy = py::module_::import("numpy");
    py::object inertia_a = numpy.attr("diag")(py::make_tuple(1.0, 2.0, 3.0));
    py::object inertia_b = numpy.attr("diag")(py::make_tuple(1.0, 2.0, 3.0));
    py::list panels_a; panels_a.append(panel_a);
    py::list panels_b; panels_b.append(panel_b);
    py::object geometry_a = module.attr("SatelliteGeometry")(3.0, inertia_a, panels_a);
    py::object geometry_b = module.attr("SatelliteGeometry")(3.0, inertia_b, panels_b);
    CHECK(geometry_a.equal(geometry_b));
    CHECK(py::str(geometry_a).cast<std::string>().find("SatelliteGeometry") != std::string::npos);

    bool panel_args_rejected = false;
    try { (void)module.attr("SatellitePanel")(py::make_tuple(0, 0, 0), py::make_tuple(1, 0, 0)); }
    catch (const py::error_already_set&) { panel_args_rejected = true; PyErr_Clear(); }
    CHECK(panel_args_rejected);
    bool geometry_args_rejected = false;
    try { (void)module.attr("SatelliteGeometry")(py::arg("inertia") = inertia_a); }
    catch (const py::error_already_set&) { geometry_args_rejected = true; PyErr_Clear(); }
    CHECK(geometry_args_rejected);

    bool extra_attr_rejected = false;
    try { geometry_a.attr("name") = "spacecraft"; }
    catch (const py::error_already_set&) { extra_attr_rejected = true; PyErr_Clear(); }
    CHECK(extra_attr_rejected);
}

int main(int argc, char** argv) {
    py::scoped_interpreter interpreter{};
    CHECK(argc == 2);
    py::module_::import("sys").attr("path").attr("insert")(0, GENADCS_PYTHON_SITE_PACKAGES);
    const py::module_ module = geometry_module(argv[1]);
    test_geometry_defaults_mutation_and_independent_lists(module);
    test_geometry_retains_constructor_inputs_and_panel_list_alias(module);
    test_geometry_value_equality_repr_slots_and_argument_errors(module);
}
