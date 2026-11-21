pub struct NativeEphemeris {
    kernel: Kernel,
}
impl NativeEphemeris {
    pub fn from_path(path: &Path) -> PyResult<Self> {
        Ok(Self {
            kernel: Kernel::load(path)?,
        })
    }
    pub fn position_km(&self, body_id: i32, jd_tdb: f64) -> PyResult<[f64; 3]> {
        let index = match body_id {
            0 => 10,
            1 => 11,
            2 => 9,
            _ => {
                return Err(PyValueError::new_err(
                    "body_id must be BODY_SUN, BODY_EARTH, or BODY_MOON",
                ))
            }
        };
        Ok(self.kernel.planetary_state(index, jd_tdb)?.position)
    }
}
pub fn native_eclipse(
    r: [f64; 3],
    sun: [f64; 3],
    body_radius: f64,
    sun_radius: f64,
) -> (f64, bool, bool) {
    eclipse_calc(r, sun, body_radius, sun_radius)
}
