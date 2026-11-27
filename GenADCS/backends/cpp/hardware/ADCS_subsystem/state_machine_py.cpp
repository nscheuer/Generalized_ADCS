#include "state_machine.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace hw = omni::hardware;

namespace {

hw::DeviceMaskSpec parse_device_mask(const py::object &obj) {
  hw::DeviceMaskSpec spec;
  if (obj.is_none()) {
    return spec;
  }
  py::dict d = obj.cast<py::dict>();
  for (auto item : d) {
    spec[py::str(item.first).cast<std::string>()] = item.second.cast<bool>();
  }
  return spec;
}

std::vector<hw::ADCSTimelineEntry> parse_adcs_timeline(const py::object &obj) {
  std::vector<hw::ADCSTimelineEntry> out;
  if (obj.is_none()) {
    return out;
  }
  for (py::handle item : obj) {
    py::tuple tup = item.cast<py::tuple>();
    if (tup.size() != 2) {
      throw std::invalid_argument(
          "ADCS timeline entries must be (time, mode_or_goal)");
    }
    const double t = tup[0].cast<double>();
    if (py::isinstance<py::str>(tup[1])) {
      out.emplace_back(t, tup[1].cast<std::string>());
    } else if (py::isinstance<hw::goals::Goal>(tup[1])) {
      out.emplace_back(t, tup[1].cast<std::shared_ptr<hw::goals::Goal>>());
    } else {
      throw std::invalid_argument(
          "ADCS timeline target must be a mode name or Goal");
    }
  }
  return out;
}

} // namespace

void bind_adcs_state_machine(py::module_ &module) {
  py::class_<hw::ADCSMode>(module, "ADCSMode")
      .def(py::init([](std::string name, std::shared_ptr<hw::goals::Goal> goal,
                       const py::object &controller, const py::object &sensors,
                       const py::object &actuators) {
             std::shared_ptr<hw::controllers::Controller> ctrl;
             if (controller.is_none()) {
               ctrl = std::make_shared<hw::controllers::NoController>();
             } else if (py::isinstance<hw::controllers::Controller>(
                            controller)) {
               ctrl = controller
                          .cast<std::shared_ptr<hw::controllers::Controller>>();
             } else {
               throw py::type_error("ADCSMode controller must be a Controller "
                                    "instance");
             }
             return hw::ADCSMode(std::move(name), std::move(goal),
                                 std::move(ctrl), parse_device_mask(sensors),
                                 parse_device_mask(actuators));
           }),
           py::arg("name"), py::arg("goal") = nullptr,
           py::arg("controller") = nullptr, py::arg("sensors") = py::none(),
           py::arg("actuators") = py::none())
      .def_readwrite("name", &hw::ADCSMode::name)
      .def_readwrite("goal", &hw::ADCSMode::goal)
      .def_readwrite("controller", &hw::ADCSMode::controller)
      .def_readonly("sensors", &hw::ADCSMode::sensors)
      .def_readonly("actuators", &hw::ADCSMode::actuators);

  py::class_<hw::ADCSStateMachine, std::shared_ptr<hw::ADCSStateMachine>>(
      module, "ADCSStateMachine")
      .def(py::init([](std::string initial, std::vector<hw::ADCSMode> modes,
                       const py::object &timeline) {
             return std::make_shared<hw::ADCSStateMachine>(
                 std::move(initial), std::move(modes),
                 parse_adcs_timeline(timeline));
           }),
           py::arg("initial"), py::arg("modes"),
           py::arg("timeline") = py::none())
      .def_readwrite("initial", &hw::ADCSStateMachine::initial)
      .def_readwrite("modes", &hw::ADCSStateMachine::modes)
      .def_readonly("timeline", &hw::ADCSStateMachine::timeline);
}
