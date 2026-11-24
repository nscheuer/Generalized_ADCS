#include "conversions.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace timeconv = omni::time;

void bind_time_conversions(py::module_& module) {
    module.def("calendar_to_jd", &timeconv::calendar_to_jd,
        py::arg("year"), py::arg("month"), py::arg("day"), py::arg("hour") = 0,
        py::arg("minute") = 0, py::arg("second") = 0.0);
    module.def("jd_to_calendar", &timeconv::jd_to_calendar);
    module.def("utc_jd_to_tai_jd", &timeconv::utc_jd_to_tai_jd);
    module.def("tai_jd_to_utc_jd", &timeconv::tai_jd_to_utc_jd);
    module.def("tai_jd_to_tt_jd", &timeconv::tai_jd_to_tt_jd);
    module.def("tt_jd_to_tai_jd", &timeconv::tt_jd_to_tai_jd);
    module.def("utc_jd_to_tt_jd", &timeconv::utc_jd_to_tt_jd);
    module.def("tt_jd_to_utc_jd", &timeconv::tt_jd_to_utc_jd);
    module.def("utc_jd_to_ut1_jd", &timeconv::utc_jd_to_ut1_jd);
    module.def("ut1_jd_to_utc_jd", &timeconv::ut1_jd_to_utc_jd);
    module.def("tdb_jd_to_tt_jd", &timeconv::tdb_jd_to_tt_jd);
    module.def("utc_jd_to_tdb_jd", &timeconv::utc_jd_to_tdb_jd);
    module.def("tdb_jd_to_utc_jd", &timeconv::tdb_jd_to_utc_jd);
    module.def("tai_jd_to_ut1_jd", &timeconv::tai_jd_to_ut1_jd);
    module.def("ut1_jd_to_tai_jd", &timeconv::ut1_jd_to_tai_jd);
    module.def("tt_jd_to_ut1_jd", &timeconv::tt_jd_to_ut1_jd);
    module.def("ut1_jd_to_tt_jd", &timeconv::ut1_jd_to_tt_jd);
    module.def("tt_jd_to_tdb_jd", &timeconv::tt_jd_to_tdb_jd);
    module.def("tai_jd_to_tdb_jd", &timeconv::tai_jd_to_tdb_jd);
    module.def("tdb_jd_to_tai_jd", &timeconv::tdb_jd_to_tai_jd);
    module.def("ut1_jd_to_tdb_jd", &timeconv::ut1_jd_to_tdb_jd);
    module.def("tdb_jd_to_ut1_jd", &timeconv::tdb_jd_to_ut1_jd);
    module.def("jd_to_utc_calendar_fields", &timeconv::jd_to_utc_calendar_fields);
    module.def("day_of_year", &timeconv::day_of_year);

}
