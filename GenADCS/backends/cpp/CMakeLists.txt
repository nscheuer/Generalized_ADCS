cmake_minimum_required(VERSION 3.18)
project(GenADCS_CPP LANGUAGES C CXX)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

find_package(pybind11 CONFIG REQUIRED)

find_program(CCACHE_EXECUTABLE NAMES ccache)
if(CCACHE_EXECUTABLE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}" CACHE STRING "Compiler launcher")
    message(STATUS "Using ccache: ${CCACHE_EXECUTABLE}")
else()
    message(STATUS "ccache not found; compiling without a compiler cache")
endif()
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

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
set_target_properties(_state PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

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
)
target_compile_features(_adcs_subsystem PRIVATE cxx_std_17)
target_include_directories(_adcs_subsystem PRIVATE hardware/ADCS_subsystem)
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
set_target_properties(_coupled PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/propagation/coupled_propagators")

pybind11_add_module(_solver solver/bindings.cpp solver/coupled_solver.cpp solver/solver.cpp)
set_target_properties(_solver PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/solver")

pybind11_add_module(_simulate simulate_bindings.cpp simulate.cpp)
set_target_properties(_simulate PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

pybind11_add_module(_satellite hardware/satellite_bindings.cpp hardware/satellite.cpp)
set_target_properties(_satellite PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/hardware")
