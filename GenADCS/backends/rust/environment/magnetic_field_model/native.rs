pub enum NativeMagneticField {
    Igrf(Igrf14),
    Chaos(Chaos8),
}
pub fn native_rotation(utc_jd: f64) -> Result<[[f64; 3]; 3], String> {
    if !utc_jd.is_finite() {
        return Err("UTC Julian date must be finite".to_owned());
    }
    let tt = if utc_jd >= 2441317.5 {
        tt_jd(utc_jd).map_err(|error| error.to_string())?
    } else {
        utc_jd
    };
    Ok(gcrf_ecef(utc_jd, tt))
}
pub fn native_tdb_jd(utc_jd: f64) -> Result<f64, String> {
    if !utc_jd.is_finite() {
        return Err("UTC Julian date must be finite".to_owned());
    }
    let tt = tt_jd(utc_jd).map_err(|error| error.to_string())?;
    let g = (357.53 + 0.9856003 * (tt - 2451545.0)).to_radians();
    Ok(tt + (0.001657 * g.sin() + 0.000022 * (2.0 * g).sin()) / 86400.0)
}
impl NativeMagneticField {
    pub fn igrf14() -> Result<Self, String> {
        Ok(Self::Igrf(Igrf14::new()?))
    }
    pub fn chaos8(include_static: bool) -> Result<Self, String> {
        Ok(Self::Chaos(Chaos8::new(include_static)?))
    }
    pub fn field_inertial_tesla(&self, position: [f64; 3], jd: f64) -> Result<[f64; 3], String> {
        match self {
            Self::Igrf(model) => model.field_inertial_tesla(position, jd),
            Self::Chaos(model) => model.field_inertial_tesla(position, jd),
        }
    }
}
