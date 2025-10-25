cmake_minimum_required(VERSION 3.16)

# Disable MSYS sh interference
set(CMAKE_SH "CMAKE_SH-NOTFOUND" CACHE STRING "Disable MSYS sh" FORCE)

project(trajectory_planner)

# Prevent in-source builds
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please use a 'build' folder.")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_SHARED_LIBS ON)

# --- Find the correct Python first ---
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development.Module Development.Embed)

message(STATUS "Using Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Using Python include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "Using Python library: ${Python3_LIBRARIES}")

# --- Prepare pybind11 to use our Python and skip its own detection ---
set(PYBIND11_FINDPYTHON ON CACHE BOOL "Force pybind11 to use CMake's FindPython3 results" FORCE)
set(PYBIND11_PYTHON_VERSION "${Python3_VERSION}" CACHE STRING "Python version for pybind11" FORCE)
set(PYBIND11_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "Python executable for pybind11" FORCE)
set(PYBIND11_PYTHON_INCLUDE_DIR "${Python3_INCLUDE_DIRS}" CACHE PATH "Python includes for pybind11" FORCE)
set(PYBIND11_PYTHON_LIBRARIES "${Python3_LIBRARIES}" CACHE FILEPATH "Python libraries for pybind11" FORCE)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
)
# Pass our Python hints into pybind11 before it runs
set(FETCHCONTENT_UPDATES_DISCONNECTED TRUE)
FetchContent_MakeAvailable(pybind11)

# --- Catch2 (optional) ---
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# --- Armadillo ---
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_library(tp_nopy STATIC
			src/ArmaCSV.cpp
      src/planner/GeneralUtil.cpp
      src/planner/PlannerUtil.cpp
      src/planner/Satellite.cpp
      src/planner/OldPlanner.cpp
			)
set_target_properties(tp_nopy PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(tp STATIC
      src/planner/Satellite.cpp
      src/planner/GeneralUtil.cpp
      src/planner/OldPlanner.cpp
      src/planner/PlannerUtilPy.cpp
      src/planner/PlannerUtil.cpp
      src/ArmaNumpy.cpp
      )
set_target_properties(tp PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(sat STATIC
      src/planner/Satellite.cpp
      src/planner/GeneralUtil.cpp
      src/ArmaNumpy.cpp
      )
set_target_properties(sat PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(sat_nopy STATIC
      src/planner/Satellite.cpp
      src/planner/GeneralUtil.cpp
      )
set_target_properties(sat_nopy PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(tp_test
			   src/planner/main.cpp
			   )
set_target_properties(tp_test PROPERTIES POSITION_INDEPENDENT_CODE ON)


pybind11_add_module(tplaunch MODULE src/planner/PyPlanner.cpp)
pybind11_add_module(pysat MODULE src/planner/Satellite.cpp)


target_link_libraries(tp_nopy PUBLIC ${ARMADILLO_LIBRARIES} ${ARMADILLO_LIBRARY} pybind11::module ${PYBIND11_INCLUDE_DIRS} ${PYTHON_LIBRARY} ${PYTHON_LIBRARIES}  )
target_link_libraries(sat PUBLIC ${ARMADILLO_LIBRARIES} ${ARMADILLO_LIBRARY} pybind11::module ${PYBIND11_INCLUDE_DIRS} ${PYTHON_LIBRARY} ${PYTHON_LIBRARIES}  )
target_link_libraries(sat_nopy PUBLIC ${ARMADILLO_LIBRARIES} ${ARMADILLO_LIBRARY} pybind11::module ${PYBIND11_INCLUDE_DIRS}  ${PYTHON_LIBRARY} ${PYTHON_LIBRARIES} )
target_link_libraries(pysat PUBLIC ${ARMADILLO_LIBRARIES} ${ARMADILLO_LIBRARY} sat ${PYTHON_LIBRARY} ${PYTHON_LIBRARIES}  )
target_link_libraries(tp PUBLIC ${ARMADILLO_LIBRARIES} ${ARMADILLO_LIBRARY} pybind11::module sat ${PYBIND11_INCLUDE_DIRS} ${PYTHON_LIBRARY} ${PYTHON_LIBRARIES}   )
target_link_libraries(tplaunch PUBLIC ${ARMADILLO_LIBRARIES} ${ARMADILLO_LIBRARY} tp tp_nopy sat sat_nopy  pybind11::module ${PYBIND11_INCLUDE_DIRS} ${PYTHON_LIBRARY} ${PYTHON_LIBRARIES}  )
target_link_libraries(tp_test PRIVATE Catch2::Catch2WithMain PUBLIC pybind11::module ${PYBIND11_INCLUDE_DIRS} ${ARMADILLO_LIBRARIES} ${ARMADILLO_LIBRARY} sat tp sat_nopy tp_nopy ${PYTHON_LIBRARY} ${PYTHON_LIBRARIES})


target_include_directories(tp PUBLIC
        "${PROJECT_SOURCE_DIR}/src"
        "${PYTHON_INCLUDE_DIRS}"
        "${PYBIND11_INCLUDE_DIRS}"
        "${ARMADILLO_INCLUDE_DIRS}"
			   )

 target_include_directories(tp_nopy PUBLIC
         "${PROJECT_SOURCE_DIR}/src"
         "${PYTHON_INCLUDE_DIRS}"
         "${PYBIND11_INCLUDE_DIRS}"
         "${ARMADILLO_INCLUDE_DIRS}"
 			   )

 target_include_directories(tplaunch PUBLIC
         "${PROJECT_SOURCE_DIR}/src"
         "${PYTHON_INCLUDE_DIRS}"
         "${PYBIND11_INCLUDE_DIRS}"
         "${ARMADILLO_INCLUDE_DIRS}"
 			   )

 target_include_directories(pysat PRIVATE
         "${PROJECT_SOURCE_DIR}/src"
         "${PYTHON_INCLUDE_DIRS}"
         "${PYBIND11_INCLUDE_DIRS}"
         "${ARMADILLO_INCLUDE_DIRS}"
 			   )

 target_include_directories(sat PUBLIC
         "${PROJECT_SOURCE_DIR}/src"
         "${PYTHON_INCLUDE_DIRS}"
         "${PYBIND11_INCLUDE_DIRS}"
         "${ARMADILLO_INCLUDE_DIRS}"
 			   )


target_include_directories(sat_nopy PUBLIC
        "${PROJECT_SOURCE_DIR}/src"
        "${PYTHON_INCLUDE_DIRS}"
        "${PYBIND11_INCLUDE_DIRS}"
        "${ARMADILLO_INCLUDE_DIRS}"
			   )

target_include_directories(tp_test PUBLIC
			   "${PROJECT_SOURCE_DIR}/src"
 			   ${PROJECT_SOURCE_DIR}
         ${PYTHON_INCLUDE_DIRS}
         ${PYBIND11_INCLUDE_DIRS}
         ${ARMADILLO_INCLUDE_DIRS}
		   )
