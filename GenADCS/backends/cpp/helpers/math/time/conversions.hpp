#ifndef ADCS_OMNI_HELPERS_TIME_CONVERSIONS_HPP
#define ADCS_OMNI_HELPERS_TIME_CONVERSIONS_HPP

#include <tuple>

namespace omni::time {
double calendar_to_jd(int year, int month, int day, int hour = 0,
                      int minute = 0, double second = 0.0);
std::tuple<int, int, int, int, int, double> jd_to_calendar(double jd);
double utc_jd_to_tai_jd(double utc_jd);
double tai_jd_to_utc_jd(double tai_jd);
double tai_jd_to_tt_jd(double tai_jd);
double tt_jd_to_tai_jd(double tt_jd);
double utc_jd_to_tt_jd(double utc_jd);
double tt_jd_to_utc_jd(double tt_jd);
double utc_jd_to_ut1_jd(double utc_jd, double dut1_seconds);
double ut1_jd_to_utc_jd(double ut1_jd, double dut1_seconds);
double tdb_jd_to_tt_jd(double tdb_jd);
double utc_jd_to_tdb_jd(double utc_jd);
double tdb_jd_to_utc_jd(double tdb_jd);
double tai_jd_to_ut1_jd(double tai_jd, double dut1_seconds);
double ut1_jd_to_tai_jd(double ut1_jd, double dut1_seconds);
double tt_jd_to_ut1_jd(double tt_jd, double dut1_seconds);
double ut1_jd_to_tt_jd(double ut1_jd, double dut1_seconds);
double tt_jd_to_tdb_jd(double tt_jd);
double tai_jd_to_tdb_jd(double tai_jd);
double tdb_jd_to_tai_jd(double tdb_jd);
double ut1_jd_to_tdb_jd(double ut1_jd, double dut1_seconds);
double tdb_jd_to_ut1_jd(double tdb_jd, double dut1_seconds);
std::tuple<int, int, int, double> jd_to_utc_calendar_fields(double jd);
int day_of_year(int year, int month, int day);
} // namespace omni::time

#endif // ADCS_OMNI_HELPERS_TIME_CONVERSIONS_HPP
