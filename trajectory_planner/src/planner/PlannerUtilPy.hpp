#ifndef TPR_PLANNERUTILPY_HPP
#define TPR_PLANNERUTILPY_HPP

#include <armadillo>
#include <tuple>
#include "../ArmaNumpy.hpp"
#include <string>
#include "PlannerUtil.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;



#define DYNAMICS_INFO_PY_FORM std::tuple<py::array_t<double>,py::array_t<double>,int,py::array_t<double>,py::array_t<double>,int>

#define SYSTEM_SETTINGS_PY_FORM std::tuple<py::array_t<double>,double,double,double,double,double>
#define TRAJ_OPT_OUTPUT_PY_FORM std::tuple<int, double, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>

#define VECTOR_INFO_PY_FORM std::tuple<py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>>
#define AUGLAG_INFO_PY_FORM std::tuple<py::array_t<double>,double,py::array_t<double>>
#define TRAJECTORY_PY_FORM std::tuple<py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>>
#define OPT_PY_FORM std::tuple<py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>>
// #define OPT_TIMES_PY_FORM std::tuple<py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>>
#define AFTER_OUTPUT_PY_FORM py::tuple<int, double, OPT_PY_FORM,OPT_PY_FORM>


#define ALILQR_OUTPUT_PY_FORM std::tuple<OPT_PY_FORM, double, double>
#define BREAK_SETTINGS_PY_FORM std::tuple<int,int,int,double,double,double,int,double,double,py::array_t<double>>
#define ALILQR_SETTINGS_PY_FORM std::tuple<LINE_SEARCH_SETTINGS_FORM,AUGLAG_SETTINGS_FORM,BREAK_SETTINGS_PY_FORM,REG_SETTINGS_FORM>
#define BEFORE_OUTPUT_PY_FORM py::tuple<TRAJECTORY_PY_FORM,VECTOR_INFO_PY_FORM,COST_SETTINGS_FORM>

#define BACKWARD_PASS_RESULTS_PY_FORM std::tuple<py::array_t<double>,py::array_t<double>,py::array_t<double>>
#define ALL_SETTINGS_PY_FORM std::tuple<SYSTEM_SETTINGS_PY_FORM, ALILQR_SETTINGS_PY_FORM, ALILQR_SETTINGS_PY_FORM , INITIAL_TRAJ_SETTINGS_FORM , COST_SETTINGS_FORM ,COST_SETTINGS_FORM ,LQR_COST_SETTINGS_FORM>

#define COST_SETTINGS_FORM_CPP tuple<double, double, double, double, double,double,double,double>


VECTOR_INFO_FORM vecsPy2Cpp(VECTOR_INFO_PY_FORM py_vecs);
VECTOR_INFO_PY_FORM vecsCpp2Py(VECTOR_INFO_FORM py_vecs);
TRAJECTORY_PY_FORM trajCpp2Py(TRAJECTORY_FORM cpp_traj);
TRAJECTORY_FORM trajPy2Cpp(TRAJECTORY_PY_FORM py_traj);
AUGLAG_INFO_FORM auglagPy2Cpp(AUGLAG_INFO_PY_FORM py_auglag);
AUGLAG_INFO_PY_FORM auglagCpp2Py(AUGLAG_INFO_FORM cpp_auglag);
OPT_FORM optPy2Cpp(OPT_PY_FORM py_opt);
OPT_PY_FORM optCpp2Py(OPT_FORM cpp_opt);
// OPT_TIMES_FORM optTimesPy2Cpp(OPT_TIMES_PY_FORM py_opt);
// OPT_TIMES_PY_FORM optTimesCpp2Py(OPT_TIMES_FORM cpp_opt);

ALILQR_OUTPUT_FORM alilqrOutPy2Cpp(ALILQR_OUTPUT_PY_FORM aloutPy);
ALILQR_OUTPUT_PY_FORM alilqrOutCpp2Py(ALILQR_OUTPUT_FORM alout);

ALILQR_SETTINGS_FORM alilqrSettingsPy2Cpp(ALILQR_SETTINGS_PY_FORM alilqrPy);
ALILQR_SETTINGS_PY_FORM alilqrSettingsCpp2Py(ALILQR_SETTINGS_FORM alilqrCpp);

BREAK_SETTINGS_FORM breakSettingsPy2Cpp(BREAK_SETTINGS_PY_FORM breakPy);
BREAK_SETTINGS_PY_FORM breakSettingsCpp2Py(BREAK_SETTINGS_FORM breakCpp);

SYSTEM_SETTINGS_FORM systemSettingsPy2Cpp(SYSTEM_SETTINGS_PY_FORM sysPy);
SYSTEM_SETTINGS_PY_FORM systemSettingsCpp2Py(SYSTEM_SETTINGS_FORM sysCpp);

// CONSTRAINT_SETTINGS_PY_FORM constraintSettingsCpp2Py(CONSTRAINT_SETTINGS_FORM conCpp);
// CONSTRAINT_SETTINGS_FORM constraintSettingsPy2Cpp(CONSTRAINT_SETTINGS_PY_FORM conPy);

ALL_SETTINGS_FORM ParametersPy2Cpp(ALL_SETTINGS_PY_FORM allPy);
ALL_SETTINGS_PY_FORM ParametersCpp2Py(ALL_SETTINGS_FORM allCpp);
// ALL_SETTINGS_FORM ParametersPy2Cpp(py::tuple allPy);

py::tuple afterOutputCpp2Py(AFTER_OUTPUT_FORM afterCpp);

BACKWARD_PASS_RESULTS_PY_FORM bpOutCpp2Py(BACKWARD_PASS_RESULTS_FORM bpout);

BACKWARD_PASS_RESULTS_FORM bpOutPy2Cpp(BACKWARD_PASS_RESULTS_PY_FORM bpoutPy);

#endif
