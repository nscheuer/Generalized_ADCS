use super::*;

/// Native IGRF-14 magnetic field model.
pub struct Igrf14 {
    coefficients: Coefficients,
    epochs_jd: Vec<f64>,
}

impl Igrf14 {
    /// Load IGRF-14 coefficients.
    pub fn new() -> Result<Self, String> {
        let coefficients = parse_shc(
            include_str!(
                "../../../../../kernels/environment/magnetic_field_model/igrf14/IGRF14.shc"
            ),
            1,
            13,
        )
        .map_err(|error| error.to_string())?;
        let epochs_jd = coefficients
            .epochs
            .iter()
            .map(|year| calendar_jd(*year as i32, 1, 1))
            .collect();
        Ok(Self {
            coefficients,
            epochs_jd,
        })
    }

    /// Evaluate the field at an inertial position in metres and Julian date.
    pub fn field_inertial_tesla(
        &self,
        position_inertial_m: [f64; 3],
        epoch_jd: f64,
    ) -> Result<[f64; 3], String> {
        if position_inertial_m.iter().any(|value| !value.is_finite()) || !epoch_jd.is_finite() {
            return Err("position and epoch must be finite".to_owned());
        }
        if epoch_jd < self.epochs_jd[0] || epoch_jd > self.epochs_jd[self.epochs_jd.len() - 1] {
            return Err("IGRF-14 coefficients cover 1900-01-01 to 2030-01-01".to_owned());
        }

        let index = self
            .epochs_jd
            .partition_point(|epoch| *epoch <= epoch_jd)
            .saturating_sub(1)
            .min(self.epochs_jd.len() - 2);
        let weight = (epoch_jd - self.epochs_jd[index])
            / (self.epochs_jd[index + 1] - self.epochs_jd[index]);
        let (mut g, mut h) = padded(&self.coefficients, index, 13);
        let (g_next, h_next) = padded(&self.coefficients, index + 1, 13);
        for k in 0..g.len() {
            g[k] += weight * (g_next[k] - g[k]);
            h[k] += weight * (h_next[k] - h[k]);
        }

        evaluate(position_inertial_m, epoch_jd, &g, &h, 13).map_err(|error| error.to_string())
    }
}
