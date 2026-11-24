#include "python_conversions.hpp"
using namespace omni::state_bindings;

void bind_orbital_state(py::module_& m) {
    py::class_<OrbitalState>(m, "OrbitalState")
        .def(py::init([](py::object p, py::object v, double jd, py::object e) { return orbital(p, v, jd, e); }), py::arg("_position"), py::arg("_velocity"), py::arg("julian_date"), py::arg("ephemeris") = py::none())
        .def_static("from_cartesian", [](py::object p, py::object v, double jd, py::object e) { return orbital(p, v, jd, e); }, py::arg("position"), py::arg("velocity"), py::arg("julian_date"), py::arg("ephemeris") = py::none())
        .def_static("from_keplerian", [](double a, double e, double i, double node, double periapsis, double anomaly, double jd, py::object ephemeris) {
            KeplerianElements elements;
            elements << a, e, i, node, periapsis, anomaly;
            return orbital(py::none(), py::none(), jd, ephemeris, elements);
        }, py::arg("semi_major_axis"), py::arg("eccentricity"), py::arg("inclination"), py::arg("ascending_node"), py::arg("argument_of_periapsis"), py::arg("true_anomaly"), py::arg("julian_date"), py::arg("ephemeris") = py::none())
        .def_property("_position", [](const OrbitalState& s) -> py::object { return s.position ? py::object(to_array(*s.position)) : py::none(); },
                      [](OrbitalState& s, py::object v) { s.position = optional_vector(v, "position"); })
        .def_property("_velocity", [](const OrbitalState& s) -> py::object { return s.velocity ? py::object(to_array(*s.velocity)) : py::none(); },
                      [](OrbitalState& s, py::object v) { s.velocity = optional_vector(v, "velocity"); })
        .def_property_readonly("position", [](const OrbitalState& s) { return to_array(s.cartesian_position()); })
        .def_property_readonly("velocity", [](const OrbitalState& s) { return to_array(s.cartesian_velocity()); })
        .def_readwrite("julian_date", &OrbitalState::julian_date)
        .def_property("ephemeris", [](const OrbitalState& s) { return ephemeris_object(s.ephemeris); },
                      [](OrbitalState& s, py::handle value) { s.ephemeris = provider(value); })
        .def_property("_body_positions", [](const OrbitalState& s) {
            py::dict result;
            for (const auto& entry : s.body_positions) result[py::str(entry.first)] = to_array(entry.second);
            return result;
        }, [](OrbitalState& s, py::dict values) {
            s.body_positions.clear();
            for (const auto& entry : values) {
                s.body_positions.emplace(py::cast<std::string>(entry.first), fixed<3>(entry.second, "body position"));
            }
        })
        .def_property_readonly("_keplerian_elements", [](const OrbitalState& s) -> py::object { return s.keplerian_elements ? py::cast(*s.keplerian_elements) : py::none(); })
        .def("body_position", &body_position)
        .def_property_readonly("sun_position", [](OrbitalState& s) { return body_position(s, py::str("sun")); })
        .def_property_readonly("earth_position", [](OrbitalState& s) { return body_position(s, py::str("earth")); })
        .def_property_readonly("moon_position", [](OrbitalState& s) { return body_position(s, py::str("moon")); })
        .def_static("from_python", &orbital_from_python)
        .def("to_python", [](const OrbitalState& s) { return to_python(s); });


}
