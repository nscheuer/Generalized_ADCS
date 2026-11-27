#include "fdir.hpp"
#include "obc_subsystem.hpp"
#include "runtime.hpp"
#include "state_machine.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;
namespace obc = omni::hardware::obc;
namespace goals = omni::hardware::goals;

namespace {

std::optional<obc::HardwareMaskSpec> parse_mask(const py::object &obj) {
  if (obj.is_none()) {
    return std::nullopt;
  }
  obc::HardwareMaskSpec spec;
  py::dict d = obj.cast<py::dict>();
  for (auto item : d) {
    spec[py::str(item.first).cast<std::string>()] = item.second.cast<bool>();
  }
  return spec;
}

obc::ModeBindings parse_bindings(const py::object &obj) {
  obc::ModeBindings b;
  if (obj.is_none()) {
    return b;
  }
  py::dict d = obj.cast<py::dict>();
  if (d.contains("adcs")) {
    py::object val = d["adcs"];
    if (py::isinstance<py::str>(val)) {
      b.adcs = val.cast<std::string>();
    } else if (py::isinstance<goals::Goal>(val)) {
      b.adcs = val.cast<std::shared_ptr<goals::Goal>>();
    } else if (py::isinstance<py::tuple>(val)) {
      py::tuple tup = val.cast<py::tuple>();
      if (tup.size() != 2) {
        throw std::invalid_argument(
            "adcs tuple binding must be (mode_name, Goal)");
      }
      b.adcs = std::make_pair(
          tup[0].cast<std::string>(),
          tup[1].cast<std::shared_ptr<goals::Goal>>());
    } else if (!val.is_none()) {
      throw std::invalid_argument("unsupported adcs binding");
    }
  }
  if (d.contains("sensors")) {
    b.sensors = parse_mask(d["sensors"]);
  }
  if (d.contains("actuators")) {
    b.actuators = parse_mask(d["actuators"]);
  }
  return b;
}

std::vector<obc::TimelineEntry> parse_timeline(const py::object &obj) {
  std::vector<obc::TimelineEntry> out;
  if (obj.is_none()) {
    return out;
  }
  for (py::handle item : obj) {
    py::tuple tup = item.cast<py::tuple>();
    if (tup.size() != 2 && tup.size() != 3) {
      throw std::invalid_argument(
          "timeline entries must be (time, command) or (time, command, goal)");
    }
    const double t = tup[0].cast<double>();
    const std::string cmd = tup[1].cast<std::string>();
    std::shared_ptr<goals::Goal> g = nullptr;
    if (tup.size() == 3 && !tup[2].is_none()) {
      g = tup[2].cast<std::shared_ptr<goals::Goal>>();
    }
    out.emplace_back(t, cmd, std::move(g));
  }
  return out;
}

} // namespace

PYBIND11_MODULE(_obc_subsystem, module) {
  py::class_<obc::FDIRMonitor, std::shared_ptr<obc::FDIRMonitor>>(module,
                                                                 "FDIRMonitor")
      .def_property_readonly("name", &obc::FDIRMonitor::name);

  py::class_<obc::AngularRateMonitor, obc::FDIRMonitor,
             std::shared_ptr<obc::AngularRateMonitor>>(module,
                                                       "AngularRateMonitor")
      .def(py::init<double, std::string, std::optional<std::string>>(),
           py::arg("max_rate"), py::arg("name") = "angular_rate",
           py::arg("target_mode") = std::nullopt)
      .def_property_readonly("max_rate", &obc::AngularRateMonitor::max_rate)
      .def_property_readonly("target_mode",
                             &obc::AngularRateMonitor::target_mode);

  py::class_<obc::FDIR, std::shared_ptr<obc::FDIR>>(module, "FDIR")
      .def(py::init<std::vector<std::shared_ptr<obc::FDIRMonitor>>,
                    std::string>(),
           py::arg("monitors") = std::vector<std::shared_ptr<obc::FDIRMonitor>>{},
           py::arg("safe_mode") = "safe_mode")
      .def_property_readonly("monitors", &obc::FDIR::monitors)
      .def_property_readonly("safe_mode", &obc::FDIR::safe_mode);

  py::class_<obc::Trigger, std::shared_ptr<obc::Trigger>>(module, "Trigger");

  py::class_<obc::Telecommand, obc::Trigger, std::shared_ptr<obc::Telecommand>>(
      module, "Telecommand")
      .def(py::init([](std::string command,
                       std::shared_ptr<goals::Goal> goal,
                       const py::object &sensors,
                       const py::object &actuators) {
             return std::make_shared<obc::Telecommand>(
                 std::move(command), std::move(goal), parse_mask(sensors),
                 parse_mask(actuators));
           }),
           py::arg("command"), py::arg("goal") = nullptr,
           py::arg("sensors") = py::none(), py::arg("actuators") = py::none())
      .def_property_readonly("command", &obc::Telecommand::command)
      .def_property_readonly("goal", &obc::Telecommand::goal);

  py::class_<obc::InFOV, obc::Trigger, std::shared_ptr<obc::InFOV>>(module,
                                                                    "InFOV")
      .def(py::init<double, double, double, double, bool,
                    std::shared_ptr<goals::Goal>>(),
           py::arg("lat"), py::arg("lon"), py::arg("alt") = 0.0,
           py::arg("max_off_nadir_deg") = 45.0, py::arg("negate") = false,
           py::arg("goal") = nullptr)
      .def("__invert__", &obc::InFOV::inverted)
      .def_property_readonly("lat", &obc::InFOV::lat)
      .def_property_readonly("lon", &obc::InFOV::lon)
      .def_property_readonly("alt", &obc::InFOV::alt)
      .def_property_readonly("max_off_nadir_deg",
                             &obc::InFOV::max_off_nadir_deg)
      .def_property_readonly("negate", &obc::InFOV::negate)
      .def_property_readonly("goal", &obc::InFOV::goal);

  py::class_<obc::Completion, obc::Trigger, std::shared_ptr<obc::Completion>>(
      module, "Completion")
      .def(py::init<>());

  py::class_<obc::After, obc::Trigger, std::shared_ptr<obc::After>>(module,
                                                                    "After")
      .def(py::init<double>(), py::arg("duration"))
      .def_property_readonly("duration", &obc::After::duration);

  py::class_<obc::FDIRTriggered, obc::Trigger,
             std::shared_ptr<obc::FDIRTriggered>>(module, "FDIRTriggered")
      .def(py::init<std::optional<std::string>>(),
           py::arg("monitor") = std::nullopt)
      .def_property_readonly("monitor", &obc::FDIRTriggered::monitor);

  py::class_<obc::Mode>(module, "Mode")
      .def(py::init([](std::string name, std::vector<obc::Mode> submodes,
                       std::optional<std::string> initial_submode,
                       std::optional<double> duration,
                       std::optional<std::string> auto_target,
                       const py::object &bindings) {
             return obc::Mode(std::move(name), std::move(submodes),
                              std::move(initial_submode), duration,
                              std::move(auto_target), parse_bindings(bindings));
           }),
           py::arg("name"), py::arg("submodes") = std::vector<obc::Mode>{},
           py::arg("initial_submode") = std::nullopt,
           py::arg("duration") = std::nullopt,
           py::arg("auto_target") = std::nullopt,
           py::arg("bindings") = py::none())
      .def_readwrite("name", &obc::Mode::name)
      .def_readwrite("submodes", &obc::Mode::submodes)
      .def_readwrite("initial_submode", &obc::Mode::initial_submode)
      .def_readwrite("duration", &obc::Mode::duration)
      .def_readwrite("auto_target", &obc::Mode::auto_target);

  py::class_<obc::Transition>(module, "Transition")
      .def(py::init<std::string, std::string, std::shared_ptr<obc::Trigger>,
                    int>(),
           py::arg("source"), py::arg("target"), py::arg("trigger"),
           py::arg("priority") = 0)
      .def_readwrite("source", &obc::Transition::source)
      .def_readwrite("target", &obc::Transition::target)
      .def_readwrite("trigger", &obc::Transition::trigger)
      .def_readwrite("priority", &obc::Transition::priority);

  py::class_<obc::StateMachine, std::shared_ptr<obc::StateMachine>>(
      module, "StateMachine")
      .def(py::init([](std::string initial, std::vector<obc::Mode> modes,
                       std::vector<obc::Transition> transitions,
                       const py::object &timeline) {
             return std::make_shared<obc::StateMachine>(
                 std::move(initial), std::move(modes), std::move(transitions),
                 parse_timeline(timeline));
           }),
           py::arg("initial"), py::arg("modes"),
           py::arg("transitions") = std::vector<obc::Transition>{},
           py::arg("timeline") = py::none())
      .def_readwrite("initial", &obc::StateMachine::initial)
      .def_readwrite("modes", &obc::StateMachine::modes)
      .def_readwrite("transitions", &obc::StateMachine::transitions);

  py::class_<obc::OBCSubsystem, std::shared_ptr<obc::OBCSubsystem>>(
      module, "OBCSubsystem")
      .def(py::init([](std::shared_ptr<obc::StateMachine> state_machine,
                       std::shared_ptr<obc::FDIR> fdir, int N,
                       const py::object &timeline) {
             return std::make_shared<obc::OBCSubsystem>(
                 std::move(state_machine), std::move(fdir), N,
                 parse_timeline(timeline));
           }),
           py::arg("state_machine") = nullptr, py::arg("fdir") = nullptr,
           py::arg("N") = 1, py::arg("timeline") = py::none())
      .def_readwrite("state_machine", &obc::OBCSubsystem::state_machine)
      .def_readwrite("fdir", &obc::OBCSubsystem::fdir)
      .def_readwrite("N", &obc::OBCSubsystem::N);
}
