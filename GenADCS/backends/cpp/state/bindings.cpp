#include "state.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace omni;

namespace {
py::object numpy() { return py::module_::import("numpy"); }
py::object python_states() { return py::module_::import("GenADCS.backends.python.state"); }

py::array as_array(py::handle value, const char* name, int ndim, py::ssize_t size = -1) {
    py::array array = numpy().attr("asarray")(value, py::arg("dtype") = "float64").cast<py::array>();
    if (array.ndim() != ndim || (size >= 0 && array.shape(0) != size) ||
        !numpy().attr("all")(numpy().attr("isfinite")(array)).cast<bool>()) {
        throw py::value_error(std::string(name) + " must be a finite " +
                              (size >= 0 ? std::to_string(size) + "-vector" : "1D array"));
    }
    return array;
}

template <int Size>
Eigen::Matrix<double, Size, 1> fixed(py::handle value, const char* name) {
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>(
        as_array(value, name, 1, Size));
    auto data = array.unchecked<1>();
    Eigen::Matrix<double, Size, 1> result;
    for (int index = 0; index < Size; ++index) {
        result[index] = data(index);
    }
    return result;
}

class PythonEphemeris final : public EphemerisProvider {
public:
    explicit PythonEphemeris(py::object value) : model(std::move(value)) {}

    Vector3 position(int body_id, double julian_date) const override {
        py::gil_scoped_acquire gil;
        return fixed<3>(model.attr("position")(body_id, julian_date), "ephemeris position");
    }

    py::object model;
};

std::shared_ptr<const EphemerisProvider> provider(py::handle value) {
    if (value.is_none()) return nullptr;
    return std::make_shared<PythonEphemeris>(py::reinterpret_borrow<py::object>(value));
}

py::object ephemeris_object(const std::shared_ptr<const EphemerisProvider>& value) {
    if (!value) return py::none();
    const auto python = std::dynamic_pointer_cast<const PythonEphemeris>(value);
    if (!python) return py::none();
    return python->model;
}

Eigen::VectorXd variable(py::handle value, const char* name) {
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>(
        as_array(value, name, 1));
    auto data = array.unchecked<1>();
    Eigen::VectorXd result(array.size());
    for (py::ssize_t index = 0; index < array.size(); ++index) {
        result[index] = data(index);
    }
    return result;
}

template <typename Derived>
py::array_t<double> to_array(const Eigen::MatrixBase<Derived>& values) {
    const py::ssize_t count = static_cast<py::ssize_t>(values.size());
    py::array_t<double> result(count);
    auto data = result.mutable_unchecked<1>();
    for (py::ssize_t index = 0; index < count; ++index) {
        data(index) = values.derived().coeff(index);
    }
    return result;
}

py::object orientation_value(py::object value) {
    return py::hasattr(value, "hamilton_components") ? value.attr("hamilton_components") : value;
}

AttitudeState attitude_from_python(py::object state) {
    return AttitudeState(fixed<4>(orientation_value(state.attr("orientation")), "orientation"),
                         fixed<3>(state.attr("rates"), "rates"));
}

std::optional<Vector3> optional_vector(py::handle value, const char* name) {
    if (value.is_none()) return std::nullopt;
    return fixed<3>(value, name);
}

OrbitalState orbital(py::object position, py::object velocity, double jd, py::object ephemeris,
                     std::optional<KeplerianElements> elements = std::nullopt) {
    return OrbitalState(optional_vector(position, "position"), optional_vector(velocity, "velocity"),
                        jd, elements, provider(ephemeris));
}

OrbitalState orbital_from_python(py::object state) {
    py::object elements = state.attr("_keplerian_elements");
    if (elements.is_none()) {
        return orbital(state.attr("_position"), state.attr("_velocity"),
                       state.attr("julian_date").cast<double>(), state.attr("ephemeris"));
    }
    return orbital(py::none(), py::none(), state.attr("julian_date").cast<double>(),
                   state.attr("ephemeris"), elements.cast<KeplerianElements>());
}

py::object body_position(OrbitalState& orbit, py::object body) {
    const std::string key = py::str(py::type::of(body)).cast<std::string>() + ":" + py::str(body).cast<std::string>();
    int body_id;
    if (py::isinstance<py::str>(body)) {
        const std::string name = py::str(body.attr("lower")());
        if (name == "sun") body_id = 0;
        else if (name == "earth") body_id = 1;
        else if (name == "moon") body_id = 2;
        else throw py::key_error(name);
    } else body_id = body.cast<int>();
    try {
        return to_array(orbit.body_position(key, body_id));
    } catch (const std::logic_error& error) {
        if (!orbit.ephemeris) throw py::value_error(error.what());
        throw;
    }
}

py::object to_python(const AttitudeState& state) {
    return python_states().attr("AttitudeState")(to_array(state.orientation), to_array(state.rates));
}
py::object to_python(const InternalState& state) {
    return python_states().attr("InternalState")(to_array(state.wheel_momentum));
}
py::object to_python(const OrbitalState& state) {
    if (state.keplerian_elements) {
        py::tuple args(8);
        for (py::ssize_t i = 0; i < 6; ++i) args[i] = (*state.keplerian_elements)[i];
        args[6] = state.julian_date;
        args[7] = ephemeris_object(state.ephemeris);
        return python_states().attr("OrbitalState").attr("from_keplerian")(*args);
    }
    return python_states().attr("OrbitalState")(state.position ? py::object(to_array(*state.position)) : py::none(),
                                                  state.velocity ? py::object(to_array(*state.velocity)) : py::none(),
                                                  state.julian_date, ephemeris_object(state.ephemeris));
}

py::object state_record(const State& state, py::object wheel_count) {
    if (!wheel_count.is_none()) {
        const long count = wheel_count.cast<long>();
        if (count < 0) throw py::value_error("wheel_count must be nonnegative");
        state.check_wheel_count(static_cast<std::size_t>(count));
    }
    const py::ssize_t count = static_cast<py::ssize_t>(state.internal.wheel_momentum.size());
    py::object np = numpy();
    py::list fields;
    for (const auto& item : {
             std::pair<const char*, py::ssize_t>{"quaternion", 4}, {"rates", 3},
             {"position", 3}, {"velocity", 3}, {"momentum", count}}) {
        fields.append(py::make_tuple(item.first, np.attr("float64"), py::make_tuple(item.second)));
    }
    py::object record = np.attr("empty")(1, py::arg("dtype") = np.attr("dtype")(fields));
    record.attr("__setitem__")("quaternion", py::make_tuple(to_array(state.attitude.quaternion_array())));
    record.attr("__setitem__")("rates", py::make_tuple(to_array(state.attitude.rates)));
    record.attr("__setitem__")("position", py::make_tuple(to_array(state.orbit.cartesian_position())));
    record.attr("__setitem__")("velocity", py::make_tuple(to_array(state.orbit.cartesian_velocity())));
    record.attr("__setitem__")("momentum", py::make_tuple(to_array(state.internal.wheel_momentum)));
    return record;
}

State from_record(py::object record, double jd, py::object ephemeris) {
    py::object value = py::isinstance<py::array>(record) ? record[py::int_(0)] : record;
    AttitudeState attitude(fixed<4>(value["quaternion"], "quaternion"), fixed<3>(value["rates"], "rates"));
    OrbitalState orbit = orbital(value["position"], value["velocity"], jd, ephemeris);
    InternalState internal(variable(value["momentum"], "wheel_momentum"));
    return State(attitude, orbit, internal);
}
}

PYBIND11_MODULE(_state, m) {
    m.doc() = "C++ ADCS Omni state types and Python conversion helpers";

    py::class_<AttitudeState>(m, "AttitudeState")
        .def(py::init([](py::object orientation, py::object rates) {
            return AttitudeState(fixed<4>(orientation_value(orientation), "orientation"), fixed<3>(rates, "rates"));
        }), py::arg("orientation"), py::arg("rates"))
        .def_property("orientation", [](const AttitudeState& s) { return to_array(s.orientation); },
                      [](AttitudeState& s, py::object v) { s = AttitudeState(fixed<4>(v, "orientation"), s.rates); })
        .def_property("rates", [](const AttitudeState& s) { return to_array(s.rates); },
                      [](AttitudeState& s, py::object v) { s.rates = fixed<3>(v, "rates"); })
        .def_property_readonly("quaternion_array", [](const AttitudeState& s) { return to_array(s.quaternion_array()); })
        .def_static("from_python", &attitude_from_python)
        .def("to_python", [](const AttitudeState& s) { return to_python(s); });

    py::class_<InternalState>(m, "InternalState")
        .def(py::init([](py::object v) { return InternalState(variable(v, "wheel_momentum")); }), py::arg("wheel_momentum") = py::list())
        .def_property("wheel_momentum", [](const InternalState& s) { return to_array(s.wheel_momentum); },
                      [](InternalState& s, py::object v) { s = InternalState(variable(v, "wheel_momentum")); })
        .def_static("from_python", [](py::object s) { return InternalState(variable(s.attr("wheel_momentum"), "wheel_momentum")); })
        .def("to_python", [](const InternalState& s) { return to_python(s); });

    py::class_<WheelMomentumState, InternalState>(m, "WheelMomentumState")
        .def(py::init([](py::object subsystem, py::object momentum) {
            if (!subsystem.is_none() && !momentum.is_none()) throw py::value_error("provide subsystem or wheel_momentum, not both");
            if (!subsystem.is_none()) {
                py::list values;
                for (py::handle wheel : subsystem.attr("reaction_wheels")) values.append(wheel.attr("h"));
                momentum = values;
            }
            return WheelMomentumState(variable(momentum.is_none() ? py::list() : momentum, "wheel_momentum"));
        }), py::arg("subsystem") = py::none(), py::kw_only(), py::arg("wheel_momentum") = py::none())
        .def_static("from_python", [](py::object s) { return WheelMomentumState(variable(s.attr("wheel_momentum"), "wheel_momentum")); })
        .def("to_python", [](const WheelMomentumState& s) {
            return python_states().attr("WheelMomentumState")(py::arg("wheel_momentum") = to_array(s.wheel_momentum));
        });

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

    py::class_<State>(m, "State")
        .def(py::init<AttitudeState, OrbitalState, InternalState>(), py::arg("attitude"), py::arg("orbit"), py::arg("internal") = InternalState{})
        .def_readwrite("attitude", &State::attitude)
        .def_readwrite("orbit", &State::orbit)
        .def_readwrite("internal", &State::internal)
        .def("to_record", &state_record, py::arg("wheel_count") = py::none())
        .def_static("from_record", &from_record, py::arg("record"), py::arg("julian_date"), py::arg("ephemeris") = py::none())
        .def_static("from_python", [](py::object s) {
            return State(attitude_from_python(s.attr("attitude")),
                         orbital_from_python(s.attr("orbit")),
                         InternalState(variable(s.attr("internal").attr("wheel_momentum"), "wheel_momentum")));
        })
        .def("to_python", [](const State& s) {
            return python_states().attr("State")(to_python(s.attitude), to_python(s.orbit), to_python(s.internal));
        });
}
