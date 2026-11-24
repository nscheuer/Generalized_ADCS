use super::*;

/// Native CHAOS-8.7 magnetic field model.
pub struct Chaos8 {
    core: Coefficients,
    static_field: Option<Coefficients>,
}

impl Chaos8 {
    /// Load CHAOS-8.7 coefficients, optionally including the static field.
    pub fn new(include_static: bool) -> Result<Self, String> {
        let core = parse_shc(
            include_str!(
                "../../../../../kernels/environment/magnetic_field_model/chaos8/CHAOS-8.7_core.shc"
            ),
            1,
            20,
        )
        .map_err(|error| error.to_string())?;
        let static_field = if include_static {
            Some(
                parse_shc(
                    include_str!("../../../../../kernels/environment/magnetic_field_model/chaos8/CHAOS-8.7_static.shc"),
                    21,
                    185,
                )
                .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
        Ok(Self { core, static_field })
    }

    /// Whether the static field coefficients are included.
    pub fn includes_static(&self) -> bool {
        self.static_field.is_some()
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
        let year = 2000.0 + (epoch_jd - 2451544.5) / 365.25;
        if year < self.core.epochs[0] || year > self.core.epochs[self.core.epochs.len() - 1] {
            return Err("Epoch outside CHAOS-8.7 core coefficient range".to_owned());
        }

        let upper = self.core.epochs.partition_point(|epoch| *epoch <= year);
        let start = ((upper.saturating_sub(1) / 5) * 5).min(self.core.epochs.len() - 6);
        let nmax = if self.static_field.is_some() { 185 } else { 20 };
        let side = nmax + 1;
        let (mut g, mut h) = if let Some(static_field) = &self.static_field {
            padded(static_field, 0, nmax)
        } else {
            (vec![0.0; side * side], vec![0.0; side * side])
        };

        for j in 0..6 {
            let mut weight = 1.0;
            for k in 0..6 {
                if j != k {
                    weight *= (year - self.core.epochs[start + k])
                        / (self.core.epochs[start + j] - self.core.epochs[start + k]);
                }
            }
            let (g_epoch, h_epoch) = padded(&self.core, start + j, nmax);
            for n in 1..=20 {
                for m in 0..=n {
                    let index = n * side + m;
                    g[index] += weight * g_epoch[index];
                    h[index] += weight * h_epoch[index];
                }
            }
        }

        evaluate(position_inertial_m, epoch_jd, &g, &h, nmax).map_err(|error| error.to_string())
    }
}
