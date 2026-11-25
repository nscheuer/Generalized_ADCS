#include "conversions.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x))                                                                  \
      throw std::runtime_error(std::string(__FILE__) + ":" +                   \
                               std::to_string(__LINE__) + " failed: " #x);     \
  } while (false)
#define NEAR(a, b, t) CHECK(std::abs((a) - (b)) <= (t))
#define REJECT(expr)                                                           \
  do {                                                                         \
    bool caught = false;                                                       \
    try {                                                                      \
      (void)(expr);                                                            \
    } catch (const std::invalid_argument &) {                                  \
      caught = true;                                                           \
    }                                                                          \
    CHECK(caught);                                                             \
  } while (false)
namespace tc = omni::time;
constexpr double secday = 86400.;

void calendar_known_dates() {
  struct Case {
    int y, m, d, h, mi;
    double s, jd;
  };
  const std::array<Case, 6> cases{{{2000, 1, 1, 12, 0, 0., 2451545.0},
                                   {1970, 1, 1, 0, 0, 0., 2440587.5},
                                   {2000, 1, 1, 0, 0, 0., 2451544.5},
                                   {2024, 2, 29, 0, 0, 0., 2460369.5},
                                   {1900, 3, 1, 0, 0, 0., 2415079.5},
                                   {2000, 3, 1, 0, 0, 0., 2451604.5}}};
  for (const auto &c : cases)
    NEAR(tc::calendar_to_jd(c.y, c.m, c.d, c.h, c.mi, c.s), c.jd, 2e-9);
}
void calendar_roundtrips_boundaries() {
  struct Case {
    int y, m, d, h, mi;
    double s;
  };
  const std::array<Case, 4> cases{{{2000, 1, 1, 0, 0, 0.},
                                   {2024, 2, 29, 23, 59, 59.25},
                                   {1969, 7, 20, 20, 17, 40.},
                                   {1582, 10, 15, 0, 0, 0.}}};
  for (const auto &c : cases) {
    const auto [y, m, d, h, mi, s] =
        tc::jd_to_calendar(tc::calendar_to_jd(c.y, c.m, c.d, c.h, c.mi, c.s));
    CHECK(y == c.y && m == c.m && d == c.d && h == c.h && mi == c.mi);
    NEAR(s, c.s, 4e-5);
  }
  const double midnight = tc::calendar_to_jd(2025, 6, 1);
  const auto [y, m, d, h, mi, s] = tc::jd_to_calendar(midnight);
  CHECK(y == 2025 && m == 6 && d == 1 && h == 0 && mi == 0);
  NEAR(s, 0., 1e-8);
  const auto [y2, m2, d2, h2, mi2, s2] = tc::jd_to_calendar(midnight + .5);
  CHECK(y2 == 2025 && m2 == 6 && d2 == 1 && h2 == 12 && mi2 == 0);
  NEAR(s2, 0., 1e-8);
  const auto [y3, m3, d3, h3, mi3, s3] = tc::jd_to_calendar(midnight + 1);
  CHECK(y3 == 2025 && m3 == 6 && d3 == 2 && h3 == 0 && mi3 == 0);
  NEAR(s3, 0., 1e-8);
  NEAR(tc::calendar_to_jd(2020, 1, 1, 24), tc::calendar_to_jd(2020, 1, 2),
       1e-12);
  NEAR(tc::calendar_to_jd(2020, 1, 1, 0, 60), tc::calendar_to_jd(2020, 1, 1, 1),
       1e-12);
  NEAR(tc::calendar_to_jd(2020, 1, 1, 0, 0, 60.),
       tc::calendar_to_jd(2020, 1, 1, 0, 1), 1e-12);
  const auto [uy, um, ud, seconds] = tc::jd_to_utc_calendar_fields(
      tc::calendar_to_jd(2024, 3, 4, 12, 34, 56.25));
  CHECK(uy == 2024 && um == 3 && ud == 4);
  NEAR(seconds, 45296.25, 4e-5);
  REJECT(tc::jd_to_calendar(NAN));
}
void day_numbers_validate_gregorian_dates() {
  struct C {
    int y, m, d, expected;
  };
  const std::array<C, 6> cases{{{2024, 1, 1, 1},
                                {2024, 2, 29, 60},
                                {2023, 12, 31, 365},
                                {2024, 12, 31, 366},
                                {2000, 3, 1, 61},
                                {1900, 3, 1, 60}}};
  for (auto c : cases)
    CHECK(tc::day_of_year(c.y, c.m, c.d) == c.expected);
  REJECT(tc::day_of_year(2023, 2, 29));
  REJECT(tc::day_of_year(2024, 13, 1));
  REJECT(tc::day_of_year(2024, 0, 1));
  REJECT(tc::day_of_year(2024, 4, 31));
}
void leap_second_and_tai_tt() {
  constexpr std::array<double, 28> effective{
      {2441317.5, 2441499.5, 2441683.5, 2442048.5, 2442413.5, 2442778.5,
       2443144.5, 2443509.5, 2443874.5, 2444239.5, 2444786.5, 2445151.5,
       2445516.5, 2446247.5, 2447161.5, 2447892.5, 2448257.5, 2448804.5,
       2449169.5, 2449534.5, 2450083.5, 2450630.5, 2451179.5, 2453736.5,
       2454832.5, 2456109.5, 2457204.5, 2457754.5}};
  for (std::size_t i = 0; i < effective.size(); ++i) {
    const double after = tc::utc_jd_to_tai_jd(effective[i]);
    NEAR((after - effective[i]) * secday, 10. + static_cast<double>(i), 2e-5);
    if (i) {
      const double before = tc::utc_jd_to_tai_jd(effective[i] - 1e-6);
      NEAR((after - before) * secday, 1.0864, 2e-4);
    }
  }
  const double utc = tc::calendar_to_jd(2024, 5, 6, 7, 8, 9.);
  NEAR(tc::tai_jd_to_utc_jd(tc::utc_jd_to_tai_jd(utc)), utc, 2e-10);
  const double tai = 2459000.123456, tt = tc::tai_jd_to_tt_jd(tai);
  NEAR((tt - tai) * secday, 32.184, 2e-5);
  NEAR(tc::tt_jd_to_tai_jd(tt), tai, 2e-10);
  REJECT(tc::utc_jd_to_tai_jd(2441317.5 - 1e-5));
  REJECT(tc::utc_jd_to_tt_jd(2440000.));
  REJECT(tc::utc_jd_to_tai_jd(NAN));
}
void ut1_tdb_and_scale_paths() {
  constexpr double utc = 2459000.25;
  for (double dut1 : {-.8, 0., .45, 1.2}) {
    const double ut1 = tc::utc_jd_to_ut1_jd(utc, dut1);
    NEAR((ut1 - utc) * secday, dut1, 2e-5);
    NEAR(tc::ut1_jd_to_utc_jd(ut1, dut1), utc, 2e-10);
  }
  const double dut1 = -.137;
  const double tai =
      tc::utc_jd_to_tai_jd(tc::calendar_to_jd(2022, 9, 17, 3, 4, 5.));
  const double tt = tc::tai_jd_to_tt_jd(tai), tdb = tc::tt_jd_to_tdb_jd(tt),
               ut1 = tc::tai_jd_to_ut1_jd(tai, dut1);
  NEAR(tc::tt_jd_to_tai_jd(tt), tai, 2e-10);
  NEAR(tc::tdb_jd_to_tt_jd(tdb), tt, 2e-10);
  NEAR(tc::tdb_jd_to_tai_jd(tdb), tai, 3e-10);
  NEAR(tc::tai_jd_to_tdb_jd(tai), tdb, 2e-10);
  NEAR(tc::ut1_jd_to_tai_jd(ut1, dut1), tai, 2e-10);
  NEAR(tc::ut1_jd_to_tt_jd(ut1, dut1), tt, 3e-10);
  NEAR(tc::tt_jd_to_ut1_jd(tt, dut1), ut1, 3e-10);
  NEAR(tc::ut1_jd_to_tdb_jd(ut1, dut1), tdb, 4e-10);
  NEAR(tc::tdb_jd_to_ut1_jd(tdb, dut1), ut1, 4e-10);
  double max_abs = 0., min = 1., max = -1., max_step = 0., previous = 0.;
  for (int i = 0; i < 100; ++i) {
    const double jd = 2451545. + 365.25 * i / 99.;
    const double corr = (tc::tt_jd_to_tdb_jd(jd) - jd) * secday;
    max_abs = std::max(max_abs, std::abs(corr));
    min = std::min(min, corr);
    max = std::max(max, corr);
    if (i)
      max_step = std::max(max_step, std::abs(corr - previous));
    previous = corr;
  }
  CHECK(max_abs < .002);
  CHECK(max - min > .002);
  CHECK(max_step < .001);
}
int main() {
  calendar_known_dates();
  calendar_roundtrips_boundaries();
  day_numbers_validate_gregorian_dates();
  leap_second_and_tai_tt();
  ut1_tdb_and_scale_paths();
}
