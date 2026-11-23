#include "check.hpp"

int main(int argc, char** argv) {
    py::scoped_interpreter interpreter{};
    CHECK(argc == 2);
    py::module_ module = load_adcs_subsystem_module(argv[1]);
    install_effect_type_stubs();
    py::object type = module.attr("ReactionWheel");

    py::object wheel = type(py::make_tuple(1, 2, 3), py::make_tuple(0, 0, 1), 0.04, 0.2, 5.0);
    CHECK(field(wheel, "position").equal(py::make_tuple(1, 2, 3)));
    CHECK(field(wheel, "orientation").equal(py::make_tuple(0, 0, 1)));
    CHECK(field(wheel, "rotational_inertia").cast<double>() == 0.04);
    CHECK(field(wheel, "max_torque").cast<double>() == 0.2);
    CHECK(field(wheel, "max_momentum").cast<double>() == 5.0);
    CHECK(field(wheel, "misalignment").equal(py::make_tuple(0.0, 0.0, 0.0)));
    CHECK(py::len(field(wheel, "effects")) == 0);
    CHECK(py::len(field(wheel, "noises")) == 0);
    CHECK(py::len(field(wheel, "biases")) == 0);

    py::object noise_type = py::module_::import("GenADCS.backends.python.noise").attr("Noise");
    py::object bias_type = py::module_::import("GenADCS.backends.python.bias").attr("Bias");
    py::object noise = noise_type();
    py::object bias = bias_type();
    py::list typed_effects; typed_effects.append(noise); typed_effects.append(bias);
    wheel.attr("effects") = typed_effects;
    CHECK(py::len(field(wheel, "noises")) == 1);
    CHECK(field(wheel, "noises")[0].is(noise));
    CHECK(py::len(field(wheel, "biases")) == 1);
    CHECK(field(wheel, "biases")[0].is(bias));

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
    py::list effects;
    effects.append(scope["Add"](-1.0, calls));
    effects.append(scope["Add"](0.25, calls));
    wheel.attr("effects") = effects;
    CHECK(wheel.attr("apply_errors")(2.0, 0.5, rng).cast<double>() == 1.25);
    CHECK(calls[0][1].cast<double>() == 0.5 && calls[0][2].is(rng));
    CHECK(calls[1][0].cast<double>() == 1.0 && calls[1][2].is(rng));

    wheel.attr("position") = py::make_tuple(4, 5, 6);
    wheel.attr("rotational_inertia") = 0.05;
    wheel.attr("max_torque") = 0.12;
    wheel.attr("max_momentum") = 2.5;
    wheel.attr("misalignment") = py::make_tuple(0.01, 0.0, -0.01);
    CHECK(field(wheel, "position").equal(py::make_tuple(4, 5, 6)));
    CHECK(field(wheel, "rotational_inertia").cast<double>() == 0.05);
    CHECK(field(wheel, "max_torque").cast<double>() == 0.12);
    CHECK(field(wheel, "max_momentum").cast<double>() == 2.5);
    CHECK(field(wheel, "misalignment").equal(py::make_tuple(0.01, 0.0, -0.01)));
}
