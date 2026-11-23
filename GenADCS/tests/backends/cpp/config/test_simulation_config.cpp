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

struct Dependencies {
    py::object satellite;
    py::object environment;
    py::object state;
    py::object propagator;
    py::object solver;
    py::object record;
};

Dependencies make_dependencies(int wheel_count = 0, double julian_date = 2451545.0) {
    py::object types = py::module_::import("types");
    py::object simple_namespace = types.attr("SimpleNamespace");
    py::list wheels;
    for (int index = 0; index < wheel_count; ++index) wheels.append(py::none());

    py::object adcs = simple_namespace(py::arg("reaction_wheels") = wheels);
    py::object satellite = simple_namespace(py::arg("adcs") = adcs);
    py::object environment = simple_namespace();
    py::list record_list;
    record_list.append("state-record");
    py::object record = record_list;

    py::dict scope;
    py::exec(
        "class State:\n"
        "    def __init__(self, record, julian_date):\n"
        "        self.record = record\n"
        "        self.orbit = __import__('types').SimpleNamespace(julian_date=julian_date)\n"
        "        self.wheel_count = None\n"
        "    def to_record(self, wheel_count):\n"
        "        self.wheel_count = wheel_count\n"
        "        return self.record\n",
        scope);
    py::object state = scope["State"](record, julian_date);
    return {satellite, environment, state, simple_namespace(), simple_namespace(), record};
}

py::object construct(const py::module_& module, const Dependencies& values, py::object backend = py::str("Python")) {
    return module.attr("SimulationConfig")(
        values.satellite, values.environment, values.state, values.propagator, values.solver, backend);
}

py::object attribute(const py::object& instance, const char* name) {
    return instance.attr(name).cast<py::object>();
}

void test_default_backend_dependencies_and_derived_snapshots(const py::module_& module) {
    const Dependencies dependencies = make_dependencies(2, 2460000.25);
    py::object config = construct(module, dependencies);

    CHECK(attribute(config, "satellite").is(dependencies.satellite));
    CHECK(attribute(config, "environment").is(dependencies.environment));
    CHECK(attribute(config, "state").is(dependencies.state));
    CHECK(attribute(config, "propagator").is(dependencies.propagator));
    CHECK(attribute(config, "solver").is(dependencies.solver));
    CHECK(attribute(config, "backend").equal(py::str("Python")));
    CHECK(attribute(dependencies.state, "wheel_count").cast<int>() == 2);
    CHECK(attribute(config, "state_record").is(dependencies.record));
    CHECK(attribute(config, "julian_date").cast<double>() == 2460000.25);
}

void test_supported_and_unsupported_backends(const py::module_& module) {
    for (const char* backend : {"Python", "C", "Rust", "JAX"}) {
        const Dependencies dependencies = make_dependencies();
        py::object config = construct(module, dependencies, py::str(backend));
        CHECK(attribute(config, "backend").equal(py::str(backend)));
    }

    py::list invalid_backends;
    invalid_backends.append(py::str(""));
    invalid_backends.append(py::str("python"));
    invalid_backends.append(py::str("CUDA"));
    invalid_backends.append(py::none());
    invalid_backends.append(py::int_(1));
    for (py::handle backend_handle : invalid_backends) {
        py::object backend = py::reinterpret_borrow<py::object>(backend_handle);
        const Dependencies dependencies = make_dependencies();
        bool rejected = false;
        try { (void)construct(module, dependencies, backend); }
        catch (const py::error_already_set& error) {
            rejected = error.matches(PyExc_ValueError);
            PyErr_Clear();
        }
        CHECK(rejected);
        CHECK(attribute(dependencies.state, "wheel_count").is_none());
    }
}

void test_constructor_arity_and_readonly_snapshot_properties(const py::module_& module) {
    const Dependencies dependencies = make_dependencies();
    py::object config = construct(module, dependencies);

    bool missing_argument_rejected = false;
    try { (void)module.attr("SimulationConfig")(); }
    catch (const py::error_already_set&) { missing_argument_rejected = true; PyErr_Clear(); }
    CHECK(missing_argument_rejected);

    bool state_record_readonly = false;
    try { config.attr("state_record") = py::list(); }
    catch (const py::error_already_set&) { state_record_readonly = true; PyErr_Clear(); }
    CHECK(state_record_readonly);

    bool julian_date_readonly = false;
    try { config.attr("julian_date") = 2452000.0; }
    catch (const py::error_already_set&) { julian_date_readonly = true; PyErr_Clear(); }
    CHECK(julian_date_readonly);

    config.attr("backend") = "Rust";
    CHECK(attribute(config, "backend").equal(py::str("Rust")));
    // Backend validation is performed during construction, not on later assignment.
    config.attr("backend") = "unsupported-after-init";
    CHECK(attribute(config, "backend").equal(py::str("unsupported-after-init")));
}

int main(int argc, char** argv) {
    py::scoped_interpreter interpreter{};
    CHECK(argc == 2);
    py::module_::import("sys").attr("path").attr("insert")(0, GENADCS_PYTHON_SITE_PACKAGES);
    const py::module_ module = config_module(argv[1]);
    test_default_backend_dependencies_and_derived_snapshots(module);
    test_supported_and_unsupported_backends(module);
    test_constructor_arity_and_readonly_snapshot_properties(module);
}
