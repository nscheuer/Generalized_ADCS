cmake_minimum_required(VERSION 3.18)
project(ADCS_Omni_C LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_state state/bindings.cpp state/state.cpp)
target_compile_features(_state PRIVATE cxx_std_17)
target_include_directories(_state PRIVATE state)
set_target_properties(_state PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
