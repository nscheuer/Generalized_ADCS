# C++ backend: cured and diseased states

## Purpose

The C++ backend must keep its production implementation independent of Python while remaining fully accessible from Python. Python support belongs at the binding boundary; it must not be a dependency of the native C++ API or native C++ tests.

## Cured state

An implementation is **cured** when its production C++ headers and implementation files use native C++ types and have no Python or pybind11 dependency. Constructors and methods accept and return native values such as scalars, standard containers, and native project types.

The implementation remains fully usable from Python through a separate pybind11 binding layer. A binding file registers the native C++ classes and functions, converts Python arguments (including NumPy arrays) into native C++ values, calls the native API, and converts results back to Python values. Python imports the resulting extension module through the package's Python interface. This preserves Python-facing behavior without putting Python types in the production C++ API.

A cured native C++ test includes the production header, constructs the production object directly, calls its native methods, and checks native return values against independently calculated expectations. CTest runs the compiled test executable directly. The test links only the implementation and native dependencies it needs, so the test can run without Python or a built Python extension.

## Diseased state

A C++ test or implementation is **diseased** when C++ reaches the production implementation through Python bindings instead of using the native C++ API, or when the production C++ API itself depends on Python.

Symptoms include:

- Embedded Python interpreters, Python imports, or pybind11 objects in native C++ tests.
- Production C++ constructors or methods accepting Python objects or handles, or returning NumPy arrays.
- Native implementation headers or source files including Python or pybind11 APIs.
- Native test executables linking against Python or `pybind11::embed`.
- CTest passing a Python extension module path to a native C++ test.
- Assertions evaluated on Python objects instead of native C++ values.

The underlying disease is architectural coupling between the C++ implementation and its Python interface. Merely adding native-value helper methods does not cure a class if its public production API still exposes Python types.

## Binding boundary

Python and pybind11 are appropriate in dedicated binding files, such as `bindings.cpp`, and in Python package wrappers. Those files may handle Python objects, NumPy conversion, Python exceptions, defaults, and Python-specific methods. They should translate inputs to native C++ types, call the native API, and translate results back. Keep this conversion logic out of production C++ headers and implementation files.

A binding or integration test may exercise Python conversion behavior. Keep it explicitly labeled and registered as a binding/integration test; do not count it as a native C++ backend test.

## Requirements for native C++ tests

1. Include the production C++ header and ordinary C++ test/support headers.
2. Construct the production type directly with its C++ constructor and call its public native API directly.
3. Compare native results with independently calculated expected values. Include boundary, invalid-input, and error cases where the API defines them.
4. Keep assertions and test helpers in C++ (for example, a small `CHECK` macro or the project's native test framework).
5. Register the executable with CTest and link only the implementation and native dependencies required by that target.
6. Keep each test's scope clear: a sensor test should exercise the sensor class API, not only a helper kernel when class behavior is under test.
7. Mirror behavior covered by corresponding Python tests where parity is required. Translate assertions and expected values into C++; do not call Python tests or Python implementations to obtain expected results.
8. Run the target through CTest after building it with the project configuration.

## Forbidden in native C++ tests

Do not include or use `pybind11/embed.h`, `pybind11/eval.h`, `pybind11/numpy.h`, `py::scoped_interpreter`, `py::object`, Python imports, `py::exec`, `Py_Initialize`, or Python extension loading. Do not link a native test executable to `pybind11::embed` or `Python::Python`. Do not pass a Python module or extension path to a native test. Native tests must still run when Python is unavailable at runtime.

## Example: native implementation with Python access

The production API and implementation use native types:

```cpp
// sensor.hpp
struct Measurement {
    std::array<double, 3> value;
};

class Sensor {
public:
    explicit Sensor(std::array<double, 3> position);
    Measurement measure(std::array<double, 3> field) const;
};
```

```cpp
// sensor.cpp
Measurement Sensor::measure(std::array<double, 3> field) const {
    // Native C++ calculation.
}
```

A separate binding exposes that same class to Python and performs conversions at the boundary:

```cpp
// sensor_bindings.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "sensor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_sensor, module) {
    py::class_<Sensor>(module, "Sensor")
        .def(py::init([](py::array_t<double> position) {
            // Validate and convert the Python input to std::array<double, 3>.
            return Sensor(/* converted position */);
        }))
        .def("measure", [](const Sensor& self, py::array_t<double> field) {
            const Measurement result = self.measure(/* converted field */);
            // Convert result.value to a NumPy array and return it.
            return /* Python-facing result */;
        });
}
```

Python imports the extension module and uses `Sensor` as its public API. pybind11 converts Python inputs to native C++ values, invokes the same C++ implementation used by native callers, then converts native results back to Python values. The production C++ class and its native tests do not need Python.

## Example: diseased native test

This is written in C++, but it tests the Python-facing route rather than the native C++ API:

```cpp
#include <pybind11/embed.h>

int main() {
    pybind11::scoped_interpreter interpreter{};
    pybind11::module_ module = pybind11::module_::import("GenADCS.backends.cpp");
    pybind11::object type = module.attr("Sensor");
    pybind11::object sensor = type(/* Python arguments */);
    pybind11::object result = sensor.attr("measure")(/* Python arguments */);
    // Assertions on Python objects...
}
```

## CMake guidance

A native test target should link the implementation and native dependencies directly:

```cmake
add_executable(cpp_test_sensor test_sensor.cpp)
target_link_libraries(cpp_test_sensor PRIVATE sensor_native_core Eigen3::Eigen)
add_test(NAME cpp_sensor COMMAND cpp_test_sensor)
```

It should not link `pybind11::embed` or `Python::Python`, depend on a Python extension module, or pass an extension path in its CTest command. Production binding modules may still be built separately for Python users.
