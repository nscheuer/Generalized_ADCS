cmake_minimum_required(VERSION 3.18)
project(GenADCS_CPP LANGUAGES C CXX)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GENADCS_STRICT_WARNINGS "Treat compiler warnings as errors" ON)
if(GENADCS_STRICT_WARNINGS)
    if(MSVC)
        add_compile_options(/W4 /WX)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        RESULT_VARIABLE PYBIND11_CMAKE_DIR_RESULT
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYBIND11_CMAKE_DIR_RESULT EQUAL 0)
        set(pybind11_DIR "${PYBIND11_CMAKE_DIR}" CACHE PATH "pybind11 CMake package directory")
    endif()
endif()

find_package(pybind11 2.12 CONFIG REQUIRED)

find_program(CCACHE_EXECUTABLE NAMES ccache)
if(CCACHE_EXECUTABLE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}" CACHE STRING "Compiler launcher")
    message(STATUS "Using ccache: ${CCACHE_EXECUTABLE}")
else()
    message(STATUS "ccache not found; compiling without a compiler cache")
endif()
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include(CTest)
if(BUILD_TESTING)
    find_package(Python COMPONENTS Interpreter Development.Embed REQUIRED)
    add_library(dynamics_native_test_core STATIC
        dynamics/drag_model/projected_area.cpp
        dynamics/gravity_gradient_model/static.cpp
        dynamics/residual_dipole_model/static.cpp
        dynamics/srp_model/projected_area.cpp
        dynamics/third_body_model/lunisolar.cpp
        dynamics/zonal_model/zonal.cpp
        dynamics/_geometry.cpp
        environment/gravity/j_gravity_model.cpp
    )
    target_compile_features(dynamics_native_test_core PUBLIC cxx_std_17)
    target_include_directories(dynamics_native_test_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(dynamics_native_test_core PUBLIC Eigen3::Eigen)
    foreach(component drag_model gravity_gradient_model residual_dipole_model srp_model third_body_model zonal_model)
        add_executable(cpp_test_dynamics_${component}
            "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/dynamics/${component}/test_$<IF:$<STREQUAL:${component},gravity_gradient_model>,static,$<IF:$<STREQUAL:${component},residual_dipole_model>,static,$<IF:$<STREQUAL:${component},third_body_model>,lunisolar,$<IF:$<STREQUAL:${component},zonal_model>,zonal,projected_area>>>>.cpp")
        target_include_directories(cpp_test_dynamics_${component} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/dynamics")
        target_link_libraries(cpp_test_dynamics_${component} PRIVATE dynamics_native_test_core)
        add_test(NAME cpp_dynamics_${component} COMMAND cpp_test_dynamics_${component})
    endforeach()
    find_program(CARGO_EXECUTABLE cargo REQUIRED)
    add_library(environment_native_test_core STATIC
        environment/native_models.cpp
        environment/atmosphere/atmospheric_model.cpp
        environment/atmosphere/NRMLSISE00/nrlmsise00.cpp
        environment/atmosphere/NRMLSISE00/_nrlmsise00_core.cpp
        environment/atmosphere/NRMLSISE00/_nrlmsise00_data.cpp
        environment/atmosphere/NRMLSISE00/nrlmsise00_core.c
        environment/ephemeris/de421/de421.cpp
        environment/ephemeris/ephemeris_model.cpp
        environment/magnetic_field_model/magnetic_field_model.cpp
        environment/magnetic_field_model/_spherical_harmonics.cpp
        environment/magnetic_field_model/igrf14/igrf14.cpp
        environment/magnetic_field_model/chaos8/chaos8.cpp
        environment/magnetic_field_model/wmm/wmm.cpp
        environment/gravity/j_gravity_model.cpp
        environment/gravity/j2.cpp
        environment/gravity/j3.cpp
        environment/gravity/j4.cpp
        environment/gravity/j5.cpp
        environment/gravity/j6.cpp
        helpers/math/frames/frames.cpp
        helpers/math/frames/conversions.cpp
        helpers/math/time/conversions.cpp
    )
    target_compile_features(environment_native_test_core PUBLIC cxx_std_17)
    target_include_directories(environment_native_test_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        environment helpers/math helpers/math/frames helpers/math/time
        environment/atmosphere/NRMLSISE00
    )
    target_link_libraries(environment_native_test_core PUBLIC Eigen3::Eigen)
    set(environment_test_cases
        atmosphere
        de421
        j2 j3 j4 j5 j6 gravity_invalid_inputs
        igrf14 chaos8 wmm shc
    )
    foreach(component IN LISTS environment_test_cases)
        if(component STREQUAL "atmosphere")
            set(test_source "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/environment/atmosphere/test_atmosphere.cpp")
        elseif(component STREQUAL "de421")
            set(test_source "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/environment/ephemeris/test_de421.cpp")
        elseif(component MATCHES "^j[2-6]$")
            set(test_source "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/environment/gravity/test_${component}.cpp")
        elseif(component STREQUAL "gravity_invalid_inputs")
            set(test_source "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/environment/gravity/test_invalid_inputs.cpp")
        elseif(component STREQUAL "shc")
            set(test_source "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/environment/magnetic_field_model/test_shc.cpp")
        else()
            set(test_source "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/environment/magnetic_field_model/test_${component}.cpp")
        endif()
        add_executable(cpp_test_environment_${component} "${test_source}")
        target_include_directories(cpp_test_environment_${component} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/environment"
        )
        target_link_libraries(cpp_test_environment_${component} PRIVATE environment_native_test_core)
        add_test(NAME cpp_environment_${component} COMMAND cpp_test_environment_${component})
    endforeach()
    add_test(NAME rust_config
        COMMAND "${CARGO_EXECUTABLE}" test --manifest-path
            "${CMAKE_CURRENT_SOURCE_DIR}/../rust/config/Cargo.toml"
    )
    set_tests_properties(rust_config PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../rust/config"
    )
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE GENADCS_PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    foreach(component satellite_geometry satellite_panel)
        add_executable(cpp_test_${component}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/hardware/satellite_geometry/test_${component}.cpp
        )
        target_link_libraries(cpp_test_${component} PRIVATE pybind11::embed Python::Python)
        target_compile_definitions(cpp_test_${component} PRIVATE
            GENADCS_PYTHON_SITE_PACKAGES="${GENADCS_PYTHON_SITE_PACKAGES}"
        )
    endforeach()
    foreach(component adcs_subsystem effects magnetorquer reaction_wheel gyroscope magnetometer accelerometer coarse_sun_sensor fine_sun_sensor fine_earth_sensor)
        if(component STREQUAL "gyroscope" OR component STREQUAL "magnetometer" OR component STREQUAL "accelerometer" OR component STREQUAL "coarse_sun_sensor" OR component STREQUAL "fine_sun_sensor" OR component STREQUAL "fine_earth_sensor")
            set(test_source "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/hardware/ADCS_subsystem/sensors/test_${component}.cpp")
        else()
            set(test_source "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/hardware/ADCS_subsystem/test_${component}.cpp")
        endif()
        add_executable(cpp_test_${component}
            "${test_source}"
        )
        if(NOT component STREQUAL "coarse_sun_sensor" AND NOT component STREQUAL "fine_sun_sensor" AND NOT component STREQUAL "magnetometer" AND NOT component STREQUAL "accelerometer" AND NOT component STREQUAL "gyroscope")
            target_link_libraries(cpp_test_${component} PRIVATE pybind11::embed Python::Python)
        endif()
        if(component STREQUAL "coarse_sun_sensor" OR component STREQUAL "magnetometer" OR component STREQUAL "accelerometer" OR component STREQUAL "fine_sun_sensor" OR component STREQUAL "fine_earth_sensor" OR component STREQUAL "gyroscope")
            target_sources(cpp_test_${component} PRIVATE
                hardware/ADCS_subsystem/sensors/kernels.cpp
                helpers/math/rotations/sensor_kernels.cpp
            )
            target_include_directories(cpp_test_${component} PRIVATE
                hardware/ADCS_subsystem/sensors helpers/math/rotations
            )
            target_link_libraries(cpp_test_${component} PRIVATE Eigen3::Eigen)
            if(component STREQUAL "fine_sun_sensor")
                target_sources(cpp_test_${component} PRIVATE
                    hardware/ADCS_subsystem/sensors/fine_sun_sensor.cpp
                    helpers/random/philox.cpp
                )
                target_include_directories(cpp_test_${component} PRIVATE helpers/random)
            endif()
            if(component STREQUAL "coarse_sun_sensor")
                target_sources(cpp_test_${component} PRIVATE
                    hardware/ADCS_subsystem/sensors/coarse_sun_sensor.cpp
                    helpers/random/philox.cpp
                )
                target_include_directories(cpp_test_${component} PRIVATE helpers/random)
            endif()
            if(component STREQUAL "magnetometer")
                target_sources(cpp_test_${component} PRIVATE
                    hardware/ADCS_subsystem/sensors/magnetometer.cpp
                    helpers/random/philox.cpp
                )
                target_include_directories(cpp_test_${component} PRIVATE helpers/random)
            endif()
            if(component STREQUAL "gyroscope")
                target_sources(cpp_test_${component} PRIVATE
                    hardware/ADCS_subsystem/sensors/gyroscope.cpp
                    helpers/random/philox.cpp
                )
                target_include_directories(cpp_test_${component} PRIVATE helpers/random)
            endif()
            if(component STREQUAL "accelerometer")
                target_sources(cpp_test_${component} PRIVATE
                    hardware/ADCS_subsystem/sensors/accelerometer.cpp
                    helpers/random/philox.cpp
                )
                target_include_directories(cpp_test_${component} PRIVATE helpers/random)
            endif()
            if(component STREQUAL "fine_earth_sensor")
                target_sources(cpp_test_${component} PRIVATE
                    hardware/ADCS_subsystem/sensors/fine_sun_sensor.cpp
                    helpers/random/philox.cpp
                )
                target_include_directories(cpp_test_${component} PRIVATE helpers/random)
            endif()
        endif()
        if(NOT component STREQUAL "coarse_sun_sensor" AND NOT component STREQUAL "fine_sun_sensor" AND NOT component STREQUAL "magnetometer" AND NOT component STREQUAL "accelerometer" AND NOT component STREQUAL "gyroscope")
            target_compile_definitions(cpp_test_${component} PRIVATE
                GENADCS_PYTHON_SITE_PACKAGES="${GENADCS_PYTHON_SITE_PACKAGES}"
                GENADCS_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../../.."
            )
        endif()
    endforeach()
    foreach(component derivatives dynamics vector)
        add_executable(cpp_test_math_${component}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/helpers/math/test_${component}.cpp
            helpers/math/${component}.cpp
        )
        if(component STREQUAL "derivatives")
            target_sources(cpp_test_math_${component} PRIVATE
                helpers/math/vector.cpp
                helpers/math/rotations/quaternion/algebra.cpp
                helpers/math/rotations/quaternion/conversions.cpp
            )
            target_include_directories(cpp_test_math_${component} PRIVATE helpers/math/rotations)
        endif()
        target_link_libraries(cpp_test_math_${component} PRIVATE Eigen3::Eigen)
        target_include_directories(cpp_test_math_${component} PRIVATE helpers/math helpers/math/rotations)
        add_test(NAME cpp_math_${component} COMMAND cpp_test_math_${component})
    endforeach()
    add_library(state_native_core STATIC
        state/attitude_state.cpp
        state/internal_state.cpp
        state/orbital_state.cpp
        state/state.cpp
    )
    target_compile_features(state_native_core PUBLIC cxx_std_17)
    target_include_directories(state_native_core PUBLIC state)
    target_link_libraries(state_native_core PUBLIC Eigen3::Eigen pybind11::embed Python::Python)
    set_target_properties(state_native_core PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    foreach(component attitude_state internal_state orbital_state state)
        add_executable(cpp_test_${component}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/state/test_${component}.cpp
        )
        target_link_libraries(cpp_test_${component} PRIVATE state_native_core)
        set_target_properties(cpp_test_${component} PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON)
        add_test(NAME cpp_${component} COMMAND cpp_test_${component})
    endforeach()
    add_library(rotations_native_core STATIC
        helpers/math/rotations/_parameter_base.cpp
        helpers/math/rotations/quaternion/algebra.cpp
        helpers/math/rotations/quaternion/conversions.cpp
        helpers/math/rotations/quaternion/interpolation.cpp
        helpers/math/rotations/quaternion/quaternion.cpp
        helpers/math/rotations/dcm/kernels.cpp
        helpers/math/rotations/dcm/dcm.cpp
        helpers/math/rotations/mrp/kernels.cpp
        helpers/math/rotations/mrp/modified_rodrigues_parameters.cpp
        helpers/math/rotations/rodrigues/kernels.cpp
        helpers/math/rotations/rodrigues/rodrigues_parameters.cpp
    )
    target_include_directories(rotations_native_core PUBLIC helpers/math/rotations)
    target_link_libraries(rotations_native_core PUBLIC Eigen3::Eigen)
    foreach(component dcm quaternion mrp rodrigues)
        add_executable(cpp_test_rotations_${component}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/helpers/math/rotations/${component}/test_${component}.cpp
        )
        target_link_libraries(cpp_test_rotations_${component} PRIVATE rotations_native_core)
        add_test(NAME cpp_rotations_${component} COMMAND cpp_test_rotations_${component})
    endforeach()
    add_executable(cpp_test_time_conversions
        ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/helpers/math/time/test_conversions.cpp
        helpers/math/time/conversions.cpp
    )
    target_include_directories(cpp_test_time_conversions PRIVATE helpers/math/time)
    add_test(NAME cpp_time_conversions COMMAND cpp_test_time_conversions)
    add_library(frames_native_core STATIC
        helpers/math/frames/frames.cpp
        helpers/math/frames/conversions.cpp
    )
    target_compile_features(frames_native_core PUBLIC cxx_std_17)
    target_include_directories(frames_native_core PUBLIC helpers/math/frames)
    target_link_libraries(frames_native_core PUBLIC Eigen3::Eigen)
    foreach(component frames conversions)
        add_executable(cpp_test_frames_${component}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/helpers/math/frames/test_${component}.cpp
        )
        target_link_libraries(cpp_test_frames_${component} PRIVATE frames_native_core)
        add_test(NAME cpp_frames_${component} COMMAND cpp_test_frames_${component})
    endforeach()
    add_executable(cpp_test_random_philox
        ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/helpers/random/test_philox.cpp
        helpers/random/philox.cpp
    )
    target_include_directories(cpp_test_random_philox PRIVATE helpers/random)
    add_test(NAME cpp_random_philox COMMAND cpp_test_random_philox)
endif()

pybind11_add_module(_bias
    bias/bindings.cpp
    bias/bias.cpp
)
target_compile_features(_bias PRIVATE cxx_std_17)
set_target_properties(_bias PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bias")

pybind11_add_module(_state
    state/bindings.cpp
    state/state.cpp
    state/attitude_state.cpp
    state/internal_state.cpp
    state/orbital_state.cpp
)
target_compile_features(_state PRIVATE cxx_std_17)
target_link_libraries(_state PRIVATE Eigen3::Eigen)
target_include_directories(_state PRIVATE state)
set_target_properties(_state PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

pybind11_add_module(_frames
    helpers/math/frames/bindings.cpp
    helpers/math/frames/frames.cpp
    helpers/math/frames/conversions.cpp
)
target_compile_features(_frames PRIVATE cxx_std_17)
target_link_libraries(_frames PRIVATE Eigen3::Eigen)
target_include_directories(_frames PRIVATE helpers/math/frames)
set_target_properties(_frames PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/helpers/math/frames")

pybind11_add_module(_helpers
    helpers/math/bindings.cpp
    helpers/math/vector.cpp
    helpers/math/derivatives.cpp
    helpers/math/dynamics.cpp
    helpers/math/rotations/_parameter_base.cpp
    helpers/math/rotations/quaternion/algebra.cpp
    helpers/math/rotations/quaternion/conversions.cpp
    helpers/math/rotations/quaternion/interpolation.cpp
    helpers/math/rotations/quaternion/quaternion.cpp
    helpers/math/rotations/dcm/kernels.cpp
    helpers/math/rotations/dcm/dcm.cpp
    helpers/math/rotations/mrp/kernels.cpp
    helpers/math/rotations/mrp/modified_rodrigues_parameters.cpp
    helpers/math/rotations/rodrigues/kernels.cpp
    helpers/math/rotations/rodrigues/rodrigues_parameters.cpp
    helpers/math/time/conversions.cpp
)
target_compile_features(_helpers PRIVATE cxx_std_17)
target_link_libraries(_helpers PRIVATE Eigen3::Eigen)
target_include_directories(_helpers PRIVATE helpers/math)
set_target_properties(_helpers PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/helpers/math")

pybind11_add_module(_adcs_subsystem
    hardware/ADCS_subsystem/bindings.cpp
    hardware/ADCS_subsystem/adcs_subsystem.cpp
    hardware/ADCS_subsystem/effects.cpp
    hardware/ADCS_subsystem/magnetorquer.cpp
    hardware/ADCS_subsystem/reaction_wheel.cpp
    hardware/ADCS_subsystem/sensors/gyroscope.cpp
    hardware/ADCS_subsystem/sensors/magnetometer.cpp
    hardware/ADCS_subsystem/sensors/kernels.cpp
    hardware/ADCS_subsystem/sensors/accelerometer.cpp
    hardware/ADCS_subsystem/sensors/coarse_sun_sensor.cpp
    hardware/ADCS_subsystem/sensors/fine_sun_sensor.cpp
    helpers/math/rotations/sensor_kernels.cpp
    helpers/math/rotations/sensor_kernels_pybind.cpp
    helpers/random/philox.cpp
)
target_compile_features(_adcs_subsystem PRIVATE cxx_std_17)
target_link_libraries(_adcs_subsystem PRIVATE Eigen3::Eigen)
target_include_directories(_adcs_subsystem PRIVATE hardware/ADCS_subsystem hardware/ADCS_subsystem/sensors helpers/random helpers/math/rotations)
set_target_properties(_adcs_subsystem PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/hardware/ADCS_subsystem")

pybind11_add_module(_satellite_geometry
    hardware/satellite_geometry/bindings.cpp
    hardware/satellite_geometry/satellite_geometry.cpp
    hardware/satellite_geometry/satellite_panel.cpp
)
target_compile_features(_satellite_geometry PRIVATE cxx_std_17)
target_include_directories(_satellite_geometry PRIVATE hardware/satellite_geometry)
set_target_properties(_satellite_geometry PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/hardware/satellite_geometry")
if(BUILD_TESTING)
    foreach(component satellite_geometry satellite_panel)
        add_dependencies(cpp_test_${component} _satellite_geometry)
        add_test(NAME cpp_${component}
            COMMAND cpp_test_${component} "$<TARGET_FILE:_satellite_geometry>"
        )
    endforeach()
    foreach(component adcs_subsystem effects magnetorquer reaction_wheel gyroscope magnetometer accelerometer coarse_sun_sensor fine_sun_sensor fine_earth_sensor)
        if(component STREQUAL "coarse_sun_sensor" OR component STREQUAL "magnetometer" OR component STREQUAL "accelerometer" OR component STREQUAL "gyroscope")
            add_test(NAME cpp_${component} COMMAND cpp_test_${component})
        elseif(NOT component STREQUAL "fine_sun_sensor" AND NOT component STREQUAL "accelerometer")
            add_dependencies(cpp_test_${component} _adcs_subsystem)
            add_test(NAME cpp_${component}
                COMMAND cpp_test_${component} "$<TARGET_FILE:_adcs_subsystem>"
            )
        else()
            add_test(NAME cpp_${component} COMMAND cpp_test_${component})
        endif()
    endforeach()
endif()

pybind11_add_module(_environment
    environment/bindings.cpp
    environment/native_models.cpp
    environment/atmosphere/NRMLSISE00/nrlmsise00.cpp
    environment/atmosphere/NRMLSISE00/_nrlmsise00_core.cpp
    environment/atmosphere/NRMLSISE00/_nrlmsise00_data.cpp
    environment/ephemeris/de421/de421.cpp
    environment/ephemeris/ephemeris_model.cpp
    environment/magnetic_field_model/magnetic_field_model.cpp
    environment/magnetic_field_model/_spherical_harmonics.cpp
    environment/magnetic_field_model/igrf14/igrf14.cpp
    environment/magnetic_field_model/chaos8/chaos8.cpp
    environment/magnetic_field_model/wmm/wmm.cpp
    environment/environment.cpp
    environment/planetary_body/planetary_body.cpp
    environment/planetary_body/earth.cpp
    environment/gravity/j_gravity_model.cpp
    environment/gravity/j2.cpp
    environment/gravity/j3.cpp
    environment/gravity/j4.cpp
    environment/gravity/j5.cpp
    environment/gravity/j6.cpp
    environment/atmosphere/atmospheric_model.cpp
    environment/atmosphere/NRMLSISE00/nrlmsise00_core.c
    helpers/math/frames/frames.cpp
    helpers/math/frames/conversions.cpp
    helpers/math/time/conversions.cpp
)
target_compile_features(_environment PRIVATE cxx_std_17)
target_link_libraries(_environment PRIVATE Eigen3::Eigen)
target_include_directories(_environment PRIVATE environment helpers/math helpers/math/frames helpers/math/time environment/atmosphere/NRMLSISE00)
set_target_properties(_environment PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/environment")

pybind11_add_module(_dynamics
    dynamics/bindings.cpp
    dynamics/_geometry.cpp
    dynamics/drag_model/projected_area.cpp
    dynamics/srp_model/projected_area.cpp
    dynamics/gravity_gradient_model/static.cpp
    dynamics/residual_dipole_model/static.cpp
    dynamics/third_body_model/lunisolar.cpp
    dynamics/zonal_model/zonal.cpp
    environment/gravity/j_gravity_model.cpp
)
target_link_libraries(_dynamics PRIVATE Eigen3::Eigen)
set_target_properties(_dynamics PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dynamics")

pybind11_add_module(_config config/bindings.cpp config/dynamics_config.cpp config/simulation_config.cpp)
target_link_libraries(_config PRIVATE Eigen3::Eigen)
set_target_properties(_config PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/config")
if(BUILD_TESTING)
    foreach(component dynamics_config simulation_config)
        add_executable(cpp_test_${component}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/backends/cpp/config/test_${component}.cpp
        )
        if(component STREQUAL "dynamics_config")
            target_include_directories(cpp_test_${component} PRIVATE config)
            add_test(NAME cpp_${component} COMMAND cpp_test_${component})
        else()
            target_link_libraries(cpp_test_${component} PRIVATE pybind11::embed Python::Python)
            target_compile_definitions(cpp_test_${component} PRIVATE
                GENADCS_PYTHON_SITE_PACKAGES="${GENADCS_PYTHON_SITE_PACKAGES}"
            )
            set_target_properties(cpp_test_${component} PROPERTIES
                CXX_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN ON)
            add_dependencies(cpp_test_${component} _config)
            add_test(NAME cpp_${component}
                COMMAND cpp_test_${component} "$<TARGET_FILE:_config>"
            )
        endif()
    endforeach()
endif()

pybind11_add_module(_coupled
    propagation/coupled_propagators/bindings.cpp
    propagation/coupled_propagators/_coupled_dynamics.cpp
    propagation/coupled_propagators/coupled_rk4_propagator.cpp
    propagation/coupled_propagators/coupled_dop853_propagator.cpp
    propagation/coupled_propagators/integrate.cpp
    dynamics/_geometry.cpp
    dynamics/drag_model/projected_area.cpp
    dynamics/srp_model/projected_area.cpp
    dynamics/gravity_gradient_model/static.cpp
    dynamics/residual_dipole_model/static.cpp
    dynamics/third_body_model/lunisolar.cpp
    environment/native_models.cpp
    environment/atmosphere/NRMLSISE00/nrlmsise00.cpp
    environment/atmosphere/NRMLSISE00/_nrlmsise00_core.cpp
    environment/atmosphere/NRMLSISE00/_nrlmsise00_data.cpp
    environment/atmosphere/NRMLSISE00/nrlmsise00_core.c
    environment/ephemeris/de421/de421.cpp
    environment/ephemeris/ephemeris_model.cpp
    environment/magnetic_field_model/magnetic_field_model.cpp
    environment/magnetic_field_model/_spherical_harmonics.cpp
    environment/magnetic_field_model/igrf14/igrf14.cpp
    environment/magnetic_field_model/chaos8/chaos8.cpp
    environment/gravity/j_gravity_model.cpp
    environment/atmosphere/atmospheric_model.cpp
    helpers/math/frames/frames.cpp
    helpers/math/frames/conversions.cpp
    helpers/math/time/conversions.cpp)
target_link_libraries(_coupled PRIVATE Eigen3::Eigen)
target_include_directories(_coupled PRIVATE environment helpers/math helpers/math/frames helpers/math/time environment/atmosphere/NRMLSISE00)
set_target_properties(_coupled PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/propagation/coupled_propagators")

pybind11_add_module(_solver solver/bindings.cpp solver/coupled_solver.cpp solver/solver.cpp)
set_target_properties(_solver PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/solver")

pybind11_add_module(_simulate simulate_bindings.cpp simulate.cpp)
set_target_properties(_simulate PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

pybind11_add_module(_satellite hardware/satellite_bindings.cpp hardware/satellite.cpp)
set_target_properties(_satellite PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/hardware")
