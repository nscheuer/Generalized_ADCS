#include "nrlmsise00.hpp"
#include "../../../helpers/math/frames/conversions.hpp"
#include "../../../helpers/math/time/conversions.hpp"
#include "../../native_helpers.hpp"
#include "_nrlmsise00_core.hpp"
#include <cmath>
namespace omni::environment {
double nrlmsise_density_geodetic(int doy, double sec, double alt, double lat,
                                 double lon, double f107, double f107a,
                                 double ap, bool effective,
                                 const std::array<double, 7> &ap_hist) {
  nrlmsise_input input{};
  input.doy = doy;
  input.sec = sec;
  input.alt = alt;
  input.g_lat = lat;
  input.g_long = lon;
  input.lst =
      std::fmod(std::fmod(sec / 3600.0 + lon / 15.0, 24.0) + 24.0, 24.0);
  input.f107A = f107a;
  input.f107 = f107;
  input.ap = ap;
  ap_array history{};
  for (std::size_t i = 0; i < 7U; ++i) {
    history.a[i] = ap_hist[i];
}
  input.ap_a = &history;
  nrlmsise_flags flags{};
  flags.switches[0] = 0;
  for (int i = 1; i < 24; ++i) {
    flags.switches[i] = 1;
}
  nrlmsise_output output{};
  run_nrlmsise00(input, flags, output, effective);
  return output.d[5] * 1000.0; // g/cm^3 to kg/m^3
}

double nrlmsise_density(const Eigen::Vector3d &p, double jd, double f107,
                        double f107a, double ap, bool effective,
                        const std::array<double, 7> &ap_hist) {
  const auto [year, month, day, seconds] =
      omni::time::jd_to_utc_calendar_fields(jd);
  const int doy = omni::time::day_of_year(year, month, day);
  const double jdtt = jd >= 2441317.5 ? omni::time::utc_jd_to_tt_jd(jd) : jd;
  const Eigen::Vector3d geodetic =
      omni::frames::gcrf_position_to_geodetic(p, jd, jdtt);
  return nrlmsise_density_geodetic(doy, seconds, geodetic.z(), geodetic.x(),
                                   geodetic.y(), f107, f107a, ap, effective,
                                   ap_hist);
}

} // namespace omni::environment
