#include <pybind11/embed.h>
#include <pybind11/eval.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;

#define CHECK(condition) do { \
    if (!(condition)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #condition); \
} while (false)

py::module_ config_module(const char* extension_path) {
    py::object util = py::module_::import("importlib.util");
    py::object spec = util.attr("spec_from_file_location")("_config", extension_path);
    py::object module = util.attr("module_from_spec")(spec);
    spec.attr("loader").attr("exec_module")(module);
    return py::reinterpret_borrow<py::module_>(module);
}

py::object attribute(const py::object& instance, const char* name) {
    return instance.attr(name).cast<py::object>();
}

bool equal(py::handle left, py::handle right) {
    return PyObject_RichCompareBool(left.ptr(), right.ptr(), Py_EQ) == 1;
}

void test_defaults_and_independent_collections(const py::module_& module) {
    py::object config_type = module.attr("DynamicsConfig");
    py::object first = config_type();
    py::object second = config_type();
    CHECK(equal(attribute(first, "forces"), py::list()));
    CHECK(equal(attribute(first, "torques"), py::list()));
    CHECK(!attribute(first, "forces").is(attribute(second, "forces")));
    CHECK(!attribute(first, "torques").is(attribute(second, "torques")));

    py::list first_forces = attribute(first, "forces");
    py::list first_torques = attribute(first, "torques");
    first_forces.append("drag");
    first_torques.append("magnetic");
    CHECK(py::len(attribute(first, "forces")) == 0);
    CHECK(py::len(attribute(second, "forces")) == 0);
    CHECK(py::len(attribute(second, "torques")) == 0);
}

void test_constructor_values_and_vector_property_updates(const py::module_& module) {
    py::list forces; forces.append("drag"); forces.append("solar-pressure");
    py::list torques; torques.append("gravity-gradient");
    py::object config = module.attr("DynamicsConfig")(forces, torques);

    // std::vector-backed pybind properties return a new Python list on each read.
    py::object exposed_forces = attribute(config, "forces");
    py::object exposed_torques = attribute(config, "torques");
    CHECK(!exposed_forces.is(forces));
    CHECK(!exposed_torques.is(torques));
    CHECK(equal(exposed_forces, forces));
    CHECK(equal(exposed_torques, torques));

    exposed_forces.attr("append")("does not update native vector");
    CHECK(py::len(attribute(config, "forces")) == 2);

    py::list replacement_forces; replacement_forces.append("gravity");
    py::list replacement_torques; replacement_torques.append("control");
    config.attr("forces") = replacement_forces;
    config.attr("torques") = replacement_torques;
    CHECK(equal(attribute(config, "forces"), replacement_forces));
    CHECK(equal(attribute(config, "torques"), replacement_torques));
}

void test_collection_type_conversion_errors(const py::module_& module) {
    bool invalid_forces = false;
    try { (void)module.attr("DynamicsConfig")(py::none(), py::list()); }
    catch (const py::error_already_set&) { invalid_forces = true; PyErr_Clear(); }
    CHECK(invalid_forces);

    bool invalid_torques = false;
    try { (void)module.attr("DynamicsConfig")(py::list(), py::none()); }
    catch (const py::error_already_set&) { invalid_torques = true; PyErr_Clear(); }
    CHECK(invalid_torques);
}

int main(int argc, char** argv) {
    py::scoped_interpreter interpreter{};
    CHECK(argc == 2);
    py::module_::import("sys").attr("path").attr("insert")(0, GENADCS_PYTHON_SITE_PACKAGES);
    const py::module_ module = config_module(argv[1]);
    test_defaults_and_independent_collections(module);
    test_constructor_values_and_vector_property_updates(module);
    test_collection_type_conversion_errors(module);
}
