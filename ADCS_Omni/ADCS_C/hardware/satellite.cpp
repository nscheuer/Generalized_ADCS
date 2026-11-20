#include "satellite.hpp"
namespace py = pybind11;
namespace omni::hardware {
Satellite::Satellite(py::object geometry_value, py::object adcs_value,
                     py::object state_machine_value)
    : geometry(geometry_value), adcs(adcs_value), state_machine(state_machine_value) {
    if (adcs.is_none())
        adcs = py::module_::import("ADCS_Omni.ADCS_C.hardware.ADCS_subsystem").attr("ADCSSubsystem")();
}
}
