use nrlmsise00::{Input, MagneticActivity, Model};

fn evaluate_geodetic(
    doy: i32,
    seconds_utc: f64,
    altitude_km: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    f107: f64,
    f107a: f64,
    ap: f64,
    ap_history: [f64; 7],
    effective_density: bool,
) -> f64 {
    let local_solar_time = (seconds_utc / 3600.0 + longitude_deg / 15.0).rem_euclid(24.0);
    let magnetic = if ap_history[0] >= 0.0 {
        MagneticActivity::History { ap: ap_history }
    } else {
        MagneticActivity::Daily { ap }
    };
    let input = Input {
        day_of_year: doy,
        seconds_of_day: seconds_utc,
        altitude_km,
        latitude_deg,
        longitude_deg,
        local_solar_time_h: local_solar_time,
        f107_average: f107a,
        f107,
        magnetic,
    };
    let model = Model::new();
    if effective_density {
        model.gtd7d(&input).total_mass
    } else {
        model.gtd7(&input).total_mass
    }
}
