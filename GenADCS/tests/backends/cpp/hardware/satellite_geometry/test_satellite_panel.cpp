#include <pybind11/embed.h>
#include <pybind11/eval.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;

#define CHECK(condition) do { \
    if (!(condition)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #condition); \
} while (false)

py::module_ panel_module(const char* extension_path) {
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

void test_panel_construction_defaults_explicit_values_and_mutability(const py::module_& module) {
    py::object panel = module.attr("SatellitePanel")(
        py::make_tuple(1.0, 2.0, 3.0),
        py::make_tuple(0.0, 0.0, 1.0),
        py::make_tuple(2.0, 1.0));
    CHECK(equal(get(panel, "position"), py::make_tuple(1.0, 2.0, 3.0)));
    CHECK(equal(get(panel, "orientation"), py::make_tuple(0.0, 0.0, 1.0)));
    CHECK(equal(get(panel, "dimensions"), py::make_tuple(2.0, 1.0)));
    CHECK(get(panel, "eta_s").cast<double>() == 0.0);
    CHECK(get(panel, "eta_d").cast<double>() == 0.0);
    CHECK(get(panel, "eta_a").cast<double>() == 1.0);
    CHECK(get(panel, "cd").cast<double>() == 2.2);

    py::object full = module.attr("SatellitePanel")(
        py::arg("position") = py::make_tuple(1, -2, 3),
        py::arg("orientation") = py::make_tuple(0.0, 0.6, 0.8),
        py::arg("dimensions") = py::make_tuple(4, 5),
        py::arg("eta_s") = 0.2, py::arg("eta_d") = 0.3,
        py::arg("eta_a") = 0.5, py::arg("cd") = 2.1);
    CHECK(equal(get(full, "position"), py::make_tuple(1, -2, 3)));
    CHECK(equal(get(full, "orientation"), py::make_tuple(0.0, 0.6, 0.8)));
    CHECK(equal(get(full, "dimensions"), py::make_tuple(4, 5)));
    CHECK(get(full, "eta_s").cast<double>() == 0.2);
    CHECK(get(full, "eta_d").cast<double>() == 0.3);
    CHECK(get(full, "eta_a").cast<double>() == 0.5);
    CHECK(get(full, "cd").cast<double>() == 2.1);

    full.attr("position") = py::make_tuple(9.0, 8.0, 7.0);
    full.attr("orientation") = py::make_tuple(1.0, 0.0, 0.0);
    full.attr("dimensions") = py::make_tuple(3.0, 4.0);
    full.attr("cd") = 2.4;
    CHECK(equal(get(full, "position"), py::make_tuple(9.0, 8.0, 7.0)));
    CHECK(equal(get(full, "orientation"), py::make_tuple(1.0, 0.0, 0.0)));
    CHECK(equal(get(full, "dimensions"), py::make_tuple(3.0, 4.0)));
    CHECK(get(full, "cd").cast<double>() == 2.4);
}

void test_panel_retains_input_objects_and_releases_them_with_instance(const py::module_& module) {
    py::dict scope;
    py::exec("class Retained: pass", scope);
    py::object retained_type = scope["Retained"];
    py::object value = retained_type();
    py::object reference = py::module_::import("weakref").attr("ref")(value);
    py::object panel = module.attr("SatellitePanel")(value, value, value);
    value = py::none();
    py::module_::import("gc").attr("collect")();
    py::object alive = reference();
    CHECK(!alive.is_none());
    CHECK(get(panel, "position").is(get(panel, "orientation")));
    CHECK(get(panel, "orientation").is(get(panel, "dimensions")));

    py::object replacement = retained_type();
    py::object replacement_ref = py::module_::import("weakref").attr("ref")(replacement);
    panel.attr("position") = replacement;
    replacement = py::none();
    py::module_::import("gc").attr("collect")();
    py::object replacement_alive = replacement_ref();
    CHECK(get(panel, "position").is(replacement_alive));
    panel = py::none();
    alive = py::none();
    replacement_alive = py::none();
    py::module_::import("gc").attr("collect")();
    py::object released = reference();
    CHECK(released.is_none());
}

void test_panel_equality_repr_slots_and_constructor_argument_errors(const py::module_& module) {
    py::object first = module.attr("SatellitePanel")(
        py::make_tuple(1, 2, 3), py::make_tuple(0, 0, 1), py::make_tuple(4, 5),
        0.1, 0.2, 0.7, 2.0);
    py::object same = module.attr("SatellitePanel")(
        py::make_tuple(1, 2, 3), py::make_tuple(0, 0, 1), py::make_tuple(4, 5),
        0.1, 0.2, 0.7, 2.0);
    CHECK(first.equal(same));
    CHECK(py::str(first).cast<std::string>().find("SatellitePanel") != std::string::npos);

    bool missing_argument_rejected = false;
    try { (void)module.attr("SatellitePanel")(py::make_tuple(0, 0, 0), py::make_tuple(1, 0, 0)); }
    catch (const py::error_already_set&) { missing_argument_rejected = true; PyErr_Clear(); }
    CHECK(missing_argument_rejected);

    bool extra_attribute_rejected = false;
    try { first.attr("metadata") = "unexpected"; }
    catch (const py::error_already_set&) { extra_attribute_rejected = true; PyErr_Clear(); }
    CHECK(extra_attribute_rejected);
}

int main(int argc, char** argv) {
    py::scoped_interpreter interpreter{};
    CHECK(argc == 2);
    py::module_::import("sys").attr("path").attr("insert")(0, GENADCS_PYTHON_SITE_PACKAGES);
    const py::module_ module = panel_module(argv[1]);
    test_panel_construction_defaults_explicit_values_and_mutability(module);
    test_panel_retains_input_objects_and_releases_them_with_instance(module);
    test_panel_equality_repr_slots_and_constructor_argument_errors(module);
}
