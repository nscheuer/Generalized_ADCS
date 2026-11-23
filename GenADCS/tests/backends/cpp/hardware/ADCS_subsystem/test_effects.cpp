#include "check.hpp"

int main(int argc, char** argv) {
    py::scoped_interpreter interpreter{};
    CHECK(argc == 2);
    py::module_ module = load_adcs_subsystem_module(argv[1]);
    py::object apply = module.attr("apply_effects");

    py::list value = py::list();
    value.append(1.0);
    value.append(2.0);
    py::object empty_result = apply(value, py::list(), 0.5);
    CHECK(empty_result.is(value));

    py::dict scope;
    py::exec(R"PY(
class Effect:
    def __init__(self, name, transform, calls):
        self.name, self.transform, self.calls = name, transform, calls
    def apply(self, value, dt, rng=None):
        self.calls.append((self.name, value, dt, rng))
        return self.transform(value)
calls = []
)PY", scope);
    py::object calls = scope["calls"];
    py::object effect_type = scope["Effect"];
    py::object rng = py::module_::import("builtins").attr("object")();
    py::list effects;
    effects.append(effect_type("double", py::cpp_function([](double x) { return x * 2.0; }), calls));
    effects.append(effect_type("offset", py::cpp_function([](double x) { return x + 3.0; }), calls));
    effects.append(effect_type("square", py::cpp_function([](double x) { return x * x; }), calls));
    CHECK(apply(4.0, effects, 0.25, rng).cast<double>() == 121.0);
    CHECK(py::len(calls) == 3);
    CHECK(item(item(calls, 0), 0).cast<std::string>() == "double");
    CHECK(item(item(calls, 0), 1).cast<double>() == 4.0);
    CHECK(item(item(calls, 1), 1).cast<double>() == 8.0);
    CHECK(item(item(calls, 2), 1).cast<double>() == 11.0);
    for (py::handle call : calls) {
        CHECK(item(call, 2).cast<double>() == 0.25);
        CHECK(item(call, 3).is(rng));
    }

    py::exec(R"PY(
class Failure:
    def apply(self, value, dt, rng=None):
        raise RuntimeError("effect failed")
class MustNotRun:
    def apply(self, value, dt, rng=None):
        raise AssertionError("later effect ran")
)PY", scope);
    py::list failing;
    failing.append(scope["Failure"]());
    failing.append(scope["MustNotRun"]());
    bool propagated = false;
    try { (void)apply(0.0, failing, 1.0); }
    catch (const py::error_already_set& error) {
        propagated = error.matches(PyExc_RuntimeError);
        PyErr_Clear();
    }
    CHECK(propagated);
}
