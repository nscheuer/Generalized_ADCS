#include "adcs_subsystem.hpp"
#include "effects.hpp"
#include "magnetorquer.hpp"
#include "reaction_wheel.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace hardware = omni::hardware;

namespace {
py::list fresh_list(const py::object& value) {
    if (value.is_none()) return py::list();
    if (PyList_Check(value.ptr()) != 0) return py::reinterpret_borrow<py::list>(value);
    return py::list(value);
}

bool same(const py::object& left, const py::object& right) {
    return py::module_::import("numpy").attr("array_equal")(left, right).cast<bool>();
}
}

PYBIND11_MODULE(_adcs_subsystem, module) {
    module.doc() = "C++ ADCS hardware subsystem containers";

    module.def("apply_effects", &hardware::apply_effects,
               py::arg("value"), py::arg("effects"), py::arg("dt"),
               py::arg("rng") = py::none());

    py::class_<hardware::ReactionWheel>(module, "ReactionWheel")
        .def(py::init([](py::object position, py::object orientation,
                         double rotational_inertia, double max_torque,
                         double max_momentum, py::object misalignment,
                         py::object effects) {
            if (misalignment.is_none()) misalignment = py::make_tuple(0.0, 0.0, 0.0);
            return hardware::ReactionWheel(std::move(position), std::move(orientation),
                rotational_inertia, max_torque, max_momentum, std::move(misalignment),
                fresh_list(effects));
        }), py::arg("position"), py::arg("orientation"),
            py::arg("rotational_inertia"), py::arg("max_torque"),
            py::arg("max_momentum"), py::arg("misalignment") = py::none(),
            py::arg("effects") = py::none())
        .def_readwrite("position", &hardware::ReactionWheel::position)
        .def_readwrite("orientation", &hardware::ReactionWheel::orientation)
        .def_readwrite("rotational_inertia", &hardware::ReactionWheel::rotational_inertia)
        .def_readwrite("max_torque", &hardware::ReactionWheel::max_torque)
        .def_readwrite("max_momentum", &hardware::ReactionWheel::max_momentum)
        .def_readwrite("misalignment", &hardware::ReactionWheel::misalignment)
        .def_readwrite("effects", &hardware::ReactionWheel::effects)
        .def_property_readonly("noises", &hardware::ReactionWheel::noises)
        .def_property_readonly("biases", &hardware::ReactionWheel::biases)
        .def("apply_errors", &hardware::ReactionWheel::apply_errors,
             py::arg("value"), py::arg("dt"), py::arg("rng") = py::none())
        .def("__eq__", [](const hardware::ReactionWheel& left,
                           const hardware::ReactionWheel& right) {
            return same(left.position, right.position) && same(left.orientation, right.orientation)
                && (left.rotational_inertia == right.rotational_inertia)
                && (left.max_torque == right.max_torque)
                && (left.max_momentum == right.max_momentum)
                && same(left.misalignment, right.misalignment) && same(left.effects, right.effects);
        });

    py::class_<hardware::Magnetorquer>(module, "Magnetorquer")
        .def(py::init([](py::object position, py::object orientation,
                         double max_dipole, py::object misalignment,
                         py::object effects) {
            if (misalignment.is_none()) misalignment = py::make_tuple(0.0, 0.0, 0.0);
            return hardware::Magnetorquer(std::move(position), std::move(orientation),
                max_dipole, std::move(misalignment), fresh_list(effects));
        }), py::arg("position"), py::arg("orientation"), py::arg("max_dipole"),
            py::arg("misalignment") = py::none(), py::arg("effects") = py::none())
        .def_readwrite("position", &hardware::Magnetorquer::position)
        .def_readwrite("orientation", &hardware::Magnetorquer::orientation)
        .def_readwrite("max_dipole", &hardware::Magnetorquer::max_dipole)
        .def_readwrite("misalignment", &hardware::Magnetorquer::misalignment)
        .def_readwrite("effects", &hardware::Magnetorquer::effects)
        .def_property_readonly("noises", &hardware::Magnetorquer::noises)
        .def_property_readonly("biases", &hardware::Magnetorquer::biases)
        .def("apply_errors", &hardware::Magnetorquer::apply_errors,
             py::arg("value"), py::arg("dt"), py::arg("rng") = py::none())
        .def("__eq__", [](const hardware::Magnetorquer& left,
                           const hardware::Magnetorquer& right) {
            return same(left.position, right.position) && same(left.orientation, right.orientation)
                && (left.max_dipole == right.max_dipole)
                && same(left.misalignment, right.misalignment) && same(left.effects, right.effects);
        });

    py::class_<hardware::ADCSSubsystem>(module, "ADCSSubsystem")
        .def(py::init([](py::object reaction_wheels, py::object magnetorquers) {
            return hardware::ADCSSubsystem(fresh_list(reaction_wheels), fresh_list(magnetorquers));
        }), py::arg("reaction_wheels") = py::none(), py::arg("magnetorquers") = py::none())
        .def_readwrite("reaction_wheels", &hardware::ADCSSubsystem::reaction_wheels)
        .def_readwrite("magnetorquers", &hardware::ADCSSubsystem::magnetorquers)
        .def("__eq__", [](const hardware::ADCSSubsystem& left,
                           const hardware::ADCSSubsystem& right) {
            return same(left.reaction_wheels, right.reaction_wheels)
                && same(left.magnetorquers, right.magnetorquers);
        });
}
