#include "check.hpp"

int main(int argc, char** argv) {
    py::scoped_interpreter interpreter{};
    CHECK(argc == 2);
    py::module_ module = load_adcs_subsystem_module(argv[1]);
    py::object subsystem_type = module.attr("ADCSSubsystem");

    py::object first = subsystem_type();
    py::object second = subsystem_type();
    CHECK(py::len(field(first, "reaction_wheels")) == 0);
    CHECK(py::len(field(first, "magnetorquers")) == 0);
    CHECK(!field(first, "reaction_wheels").is(field(second, "reaction_wheels")));
    CHECK(!field(first, "magnetorquers").is(field(second, "magnetorquers")));

    py::list wheels;
    py::list magnetorquers;
    py::object wheel = module.attr("ReactionWheel")(
        py::make_tuple(1, 2, 3), py::make_tuple(0, 0, 1), 0.05, 0.15, 3.0);
    py::object mtq = module.attr("Magnetorquer")(
        py::make_tuple(0, 1, 0), py::make_tuple(1, 0, 0), 0.25);
    wheels.append(wheel);
    magnetorquers.append(mtq);
    py::object subsystem = subsystem_type(wheels, magnetorquers);
    CHECK(field(subsystem, "reaction_wheels").is(wheels));
    CHECK(field(subsystem, "magnetorquers").is(magnetorquers));
    wheels.append("live mutation");
    CHECK(py::len(field(subsystem, "reaction_wheels")) == 2);

    py::list new_wheels;
    py::list new_mtqs;
    subsystem.attr("reaction_wheels") = new_wheels;
    subsystem.attr("magnetorquers") = new_mtqs;
    CHECK(field(subsystem, "reaction_wheels").is(new_wheels));
    CHECK(field(subsystem, "magnetorquers").is(new_mtqs));

    py::object equivalent = subsystem_type(new_wheels, new_mtqs);
    CHECK(subsystem.equal(equivalent));
    CHECK(py::str(subsystem).cast<std::string>().find("ADCSSubsystem") != std::string::npos);
    bool extra_attribute_rejected = false;
    try { subsystem.attr("name") = "extra"; }
    catch (const py::error_already_set&) { extra_attribute_rejected = true; PyErr_Clear(); }
    CHECK(extra_attribute_rejected);
}
