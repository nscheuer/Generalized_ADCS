#pragma once

#include <pybind11/embed.h>
#include <pybind11/eval.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;

#define CHECK(condition) do { \
    if (!(condition)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #condition); \
} while (false)

inline py::module_ load_adcs_subsystem_module(const char* extension_path) {
    py::module_::import("sys").attr("path").attr("insert")(0, GENADCS_PYTHON_SITE_PACKAGES);
    py::object util = py::module_::import("importlib.util");
    py::object spec = util.attr("spec_from_file_location")("_adcs_subsystem", extension_path);
    py::object module = util.attr("module_from_spec")(spec);
    spec.attr("loader").attr("exec_module")(module);
    return py::reinterpret_borrow<py::module_>(module);
}

inline py::object field(const py::object& instance, const char* name) {
    return instance.attr(name).cast<py::object>();
}

inline py::object item(py::handle sequence, Py_ssize_t index) {
    return py::reinterpret_steal<py::object>(PySequence_GetItem(sequence.ptr(), index));
}

inline void install_effect_type_stubs() {
    py::dict modules = py::module_::import("sys").attr("modules").cast<py::dict>();
    const char* parents[] = {"GenADCS", "GenADCS.backends", "GenADCS.backends.python"};
    for (const char* name : parents) {
        if (!modules.contains(name)) {
            py::module_ package(name);
            package.attr("__path__") = py::list();
            modules[name] = package;
        }
    }
    py::module_ noise("GenADCS.backends.python.noise");
    py::dict noise_scope = noise.attr("__dict__");
    py::exec("class Noise: pass", noise_scope);
    modules["GenADCS.backends.python.noise"] = noise;

    py::module_ bias("GenADCS.backends.python.bias");
    py::dict bias_scope = bias.attr("__dict__");
    py::exec("class Bias: pass", bias_scope);
    modules["GenADCS.backends.python.bias"] = bias;
}

inline py::object retained_class() {
    py::dict scope;
    py::exec("class Retained: pass", scope);
    return scope["Retained"];
}
