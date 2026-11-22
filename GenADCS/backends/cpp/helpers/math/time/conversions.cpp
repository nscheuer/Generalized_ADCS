#include "conversions.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace omni::time {
namespace {
constexpr double kSecondsPerDay = 86400.0;
constexpr double kJ2000 = 2451545.0;
constexpr std::array<double, 28> kUtcEffectiveJd{
    2441317.5, 2441499.5, 2441683.5, 2442048.5, 2442413.5, 2442778.5,
    2443144.5, 2443509.5, 2443874.5, 2444239.5, 2444786.5, 2445151.5,
    2445516.5, 2446247.5, 2447161.5, 2447892.5, 2448257.5, 2448804.5,
    2449169.5, 2449534.5, 2450083.5, 2450630.5, 2451179.5, 2453736.5,
    2454832.5, 2456109.5, 2457204.5, 2457754.5};
constexpr std::array<double, 28> kTaiMinusUtc{
    10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
    20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
    30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0};
constexpr std::array<int, 12> kMonthLengths{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

int floor_divide(int numerator, int denominator) {
    int quotient = numerator / denominator;
    const int remainder = numerator % denominator;
    if ((remainder != 0) && ((remainder < 0) != (denominator < 0))) --quotient;
    return quotient;
}

double tai_minus_utc(double utc_jd) {
    if (!std::isfinite(utc_jd) || (utc_jd < kUtcEffectiveJd.front())) {
        throw std::invalid_argument("UTC/TAI conversion is supported from 1972-01-01");
    }
    double offset = kTaiMinusUtc.front();
    for (std::size_t index = 1U; index < kUtcEffectiveJd.size(); ++index) {
        if (utc_jd < kUtcEffectiveJd[index]) break;
        offset = kTaiMinusUtc[index];
    }
    return offset;
}

double tdb_minus_tt_seconds(double tt_jd) {
    const double days = tt_jd - kJ2000;
    const double mean_anomaly = (357.53 + 0.9856003 * days) * (3.14159265358979323846 / 180.0);
    return 0.001657 * std::sin(mean_anomaly) + 0.000022 * std::sin(2.0 * mean_anomaly);
}

bool leap_year(int year) {
    return ((year % 4) == 0) && (((year % 100) != 0) || ((year % 400) == 0));
}
}

double calendar_to_jd(int year, int month, int day, int hour, int minute, double second) {
    int adjusted_year = year;
    int adjusted_month = month;
    if (adjusted_month <= 2) { --adjusted_year; adjusted_month += 12; }
    const int century = floor_divide(adjusted_year, 100);
    const int correction = 2 - century + floor_divide(century, 4);
    const double jd0 = std::floor(365.25 * static_cast<double>(adjusted_year + 4716))
        + std::floor(30.6001 * static_cast<double>(adjusted_month + 1))
        + static_cast<double>(day + correction) - 1524.5;
    const double seconds = static_cast<double>(hour * 3600 + minute * 60) + second;
    return jd0 + seconds / kSecondsPerDay;
}

std::tuple<int, int, int, int, int, double> jd_to_calendar(double jd) {
    if (!std::isfinite(jd)) throw std::invalid_argument("jd must be finite");
    const double zf = std::floor(jd + 0.5);
    const double fraction = jd + 0.5 - zf;
    const int z = static_cast<int>(zf);
    const int alpha = static_cast<int>(std::floor(static_cast<double>(z - 1867216.25) / 36524.25));
    const int a = z + 1 + alpha - floor_divide(alpha, 4);
    const int b = a + 1524;
    const int c = static_cast<int>(std::floor((static_cast<double>(b) - 122.1) / 365.25));
    const int d = static_cast<int>(std::floor(365.25 * static_cast<double>(c)));
    const int e = static_cast<int>(std::floor(static_cast<double>(b - d) / 30.6001));
    const double day_float = static_cast<double>(b - d) - std::floor(30.6001 * static_cast<double>(e)) + fraction;
    const int day = static_cast<int>(std::floor(day_float));
    const int month = (e < 14) ? (e - 1) : (e - 13);
    const int year = (month > 2) ? (c - 4716) : (c - 4715);
    double seconds = (day_float - static_cast<double>(day)) * kSecondsPerDay;
    const int hour = static_cast<int>(std::floor(seconds / 3600.0));
    seconds -= static_cast<double>(hour * 3600);
    const int minute = static_cast<int>(std::floor(seconds / 60.0));
    seconds -= static_cast<double>(minute * 60);
    return {year, month, day, hour, minute, seconds};
}

double utc_jd_to_tai_jd(double utc_jd) { return utc_jd + tai_minus_utc(utc_jd) / kSecondsPerDay; }
double tai_jd_to_utc_jd(double tai_jd) {
    double utc_jd = tai_jd - 10.0 / kSecondsPerDay;
    for (int iteration = 0; iteration < 3; ++iteration) utc_jd = tai_jd - tai_minus_utc(utc_jd) / kSecondsPerDay;
    return utc_jd;
}
double tai_jd_to_tt_jd(double tai_jd) { return tai_jd + 32.184 / kSecondsPerDay; }
double tt_jd_to_tai_jd(double tt_jd) { return tt_jd - 32.184 / kSecondsPerDay; }
double utc_jd_to_tt_jd(double utc_jd) { return tai_jd_to_tt_jd(utc_jd_to_tai_jd(utc_jd)); }
double tt_jd_to_utc_jd(double tt_jd) { return tai_jd_to_utc_jd(tt_jd_to_tai_jd(tt_jd)); }
double utc_jd_to_ut1_jd(double utc_jd, double dut1_seconds) { return utc_jd + dut1_seconds / kSecondsPerDay; }
double ut1_jd_to_utc_jd(double ut1_jd, double dut1_seconds) { return ut1_jd - dut1_seconds / kSecondsPerDay; }

double tdb_jd_to_tt_jd(double tdb_jd) {
    double tt_jd = tdb_jd;
    for (int iteration = 0; iteration < 3; ++iteration) tt_jd = tdb_jd - tdb_minus_tt_seconds(tt_jd) / kSecondsPerDay;
    return tt_jd;
}
double utc_jd_to_tdb_jd(double utc_jd) { return tt_jd_to_tdb_jd(utc_jd_to_tt_jd(utc_jd)); }
double tdb_jd_to_utc_jd(double tdb_jd) { return tt_jd_to_utc_jd(tdb_jd_to_tt_jd(tdb_jd)); }
double tai_jd_to_ut1_jd(double tai_jd, double dut1_seconds) { return utc_jd_to_ut1_jd(tai_jd_to_utc_jd(tai_jd), dut1_seconds); }
double ut1_jd_to_tai_jd(double ut1_jd, double dut1_seconds) { return utc_jd_to_tai_jd(ut1_jd_to_utc_jd(ut1_jd, dut1_seconds)); }
double tt_jd_to_ut1_jd(double tt_jd, double dut1_seconds) { return utc_jd_to_ut1_jd(tt_jd_to_utc_jd(tt_jd), dut1_seconds); }
double ut1_jd_to_tt_jd(double ut1_jd, double dut1_seconds) { return utc_jd_to_tt_jd(ut1_jd_to_utc_jd(ut1_jd, dut1_seconds)); }
double tt_jd_to_tdb_jd(double tt_jd) { return tt_jd + tdb_minus_tt_seconds(tt_jd) / kSecondsPerDay; }
double tai_jd_to_tdb_jd(double tai_jd) { return tt_jd_to_tdb_jd(tai_jd_to_tt_jd(tai_jd)); }
double tdb_jd_to_tai_jd(double tdb_jd) { return tt_jd_to_tai_jd(tdb_jd_to_tt_jd(tdb_jd)); }
double ut1_jd_to_tdb_jd(double ut1_jd, double dut1_seconds) { return utc_jd_to_tdb_jd(ut1_jd_to_utc_jd(ut1_jd, dut1_seconds)); }
double tdb_jd_to_ut1_jd(double tdb_jd, double dut1_seconds) { return utc_jd_to_ut1_jd(tdb_jd_to_utc_jd(tdb_jd), dut1_seconds); }

std::tuple<int, int, int, double> jd_to_utc_calendar_fields(double jd) {
    const auto [year, month, day, hour, minute, second] = jd_to_calendar(jd);
    return {year, month, day, static_cast<double>(hour * 3600 + minute * 60) + second};
}

int day_of_year(int year, int month, int day) {
    if ((month < 1) || (month > 12) || (day < 1) || (day > kMonthLengths[static_cast<std::size_t>(month - 1)] + ((month == 2 && leap_year(year)) ? 1 : 0))) {
        throw std::invalid_argument("invalid Gregorian calendar date");
    }
    int day_number = day;
    for (int month_index = 1; month_index < month; ++month_index) {
        day_number += kMonthLengths[static_cast<std::size_t>(month_index - 1)];
    }
    if ((month > 2) && leap_year(year)) ++day_number;
    return day_number;
}
}
