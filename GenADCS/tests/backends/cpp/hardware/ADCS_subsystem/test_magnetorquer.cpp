#include "check.hpp"

int main(int argc, char** argv) {
    py::scoped_interpreter interpreter{};
    CHECK(argc == 2);
    py::module_ module = load_adcs_subsystem_module(argv[1]);
    install_effect_type_stubs();
    py::object type = module.attr("Magnetorquer");

    py::object mtq = type(py::make_tuple(1, 2, 3), py::make_tuple(0, 0, 1), 0.25);
    CHECK(field(mtq, "position").equal(py::make_tuple(1, 2, 3)));
    CHECK(field(mtq, "orientation").equal(py::make_tuple(0, 0, 1)));
    CHECK(field(mtq, "max_dipole").cast<double>() == 0.25);
    CHECK(field(mtq, "misalignment").equal(py::make_tuple(0.0, 0.0, 0.0)));
    CHECK(py::len(field(mtq, "effects")) == 0);
    CHECK(py::len(field(mtq, "noises")) == 0);
    CHECK(py::len(field(mtq, "biases")) == 0);

    py::object noise_type = py::module_::import("GenADCS.backends.python.noise").attr("Noise");
    py::object bias_type = py::module_::import("GenADCS.backends.python.bias").attr("Bias");
    py::object noise_a = noise_type();
    py::object bias_a = bias_type();
    py::object plain = py::module_::import("builtins").attr("object")();
    py::object noise_b = noise_type();
    py::object bias_b = bias_type();
    py::list effects;
    effects.append(noise_a); effects.append(bias_a); effects.append(plain);
    effects.append(noise_b); effects.append(bias_b);
    mtq.attr("effects") = effects;
    CHECK(field(mtq, "effects").is(effects));
    CHECK(py::len(field(mtq, "noises")) == 2);
    CHECK(field(mtq, "noises")[0].is(noise_a));
    CHECK(field(mtq, "noises")[1].is(noise_b));
    CHECK(py::len(field(mtq, "biases")) == 2);
    CHECK(field(mtq, "biases")[0].is(bias_a));
    CHECK(field(mtq, "biases")[1].is(bias_b));

    py::dict scope;
    py::exec(R"PY(
class Add:
    def __init__(self, amount, calls): self.amount, self.calls = amount, calls
    def apply(self, value, dt, rng=None):
        self.calls.append((value, dt, rng))
        return value + self.amount
calls = []
)PY", scope);
    py::object calls = scope["calls"];
    py::object rng = py::module_::import("builtins").attr("object")();
    py::list apply_effects;
    apply_effects.append(scope["Add"](2.0, calls));
    apply_effects.append(scope["Add"](-0.5, calls));
    mtq.attr("effects") = apply_effects;
    CHECK(mtq.attr("apply_errors")(1.0, 0.125, rng).cast<double>() == 2.5);
    CHECK(calls[0][1].cast<double>() == 0.125 && calls[0][2].is(rng));
    CHECK(calls[1][0].cast<double>() == 3.0 && calls[1][2].is(rng));

    mtq.attr("position") = py::make_tuple(4, 5, 6);
    mtq.attr("max_dipole") = 0.5;
    mtq.attr("misalignment") = py::make_tuple(0.1, 0.0, -0.1);
    CHECK(field(mtq, "position").equal(py::make_tuple(4, 5, 6)));
    CHECK(field(mtq, "max_dipole").cast<double>() == 0.5);
    CHECK(field(mtq, "misalignment").equal(py::make_tuple(0.1, 0.0, -0.1)));
}
