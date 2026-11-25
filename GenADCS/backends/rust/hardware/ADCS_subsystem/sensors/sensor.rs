//! Rust-native common configuration for three-axis sensors.

#[derive(Clone, Debug, PartialEq)]
pub struct Sensor {
    pub position: [f64; 3],
    pub orientation: [f64; 4],
    pub misalignment: [f64; 4],
    pub saturation_value: f64,
    pub noise: [[f64; 3]; 3],
    pub bias: [f64; 3],
    pub enabled: bool,
}

impl Sensor {
    /// Construct a sensor, validating the same dimensions and saturation limit
    /// as the Python implementation. Quaternions are stored normalized.
    pub fn new(
        position: &[f64],
        orientation: &[f64],
        misalignment: &[f64],
        saturation_value: f64,
        noise: &[Vec<f64>],
        bias: &[f64],
    ) -> Result<Self, &'static str> {
        if position.len() != 3 || orientation.len() != 4 || misalignment.len() != 4 {
            return Err("position must have 3 components and quaternions 4");
        }
        if noise.len() != 3 || noise.iter().any(|row| row.len() != 3) || bias.len() != 3 {
            return Err("noise must be a 3x3 covariance and bias a 3-vector");
        }
        if saturation_value <= 0.0 {
            return Err("saturation_value must be positive");
        }
        let position = [position[0], position[1], position[2]];
        let orientation = normalize([
            orientation[0],
            orientation[1],
            orientation[2],
            orientation[3],
        ]);
        let misalignment = normalize([
            misalignment[0],
            misalignment[1],
            misalignment[2],
            misalignment[3],
        ]);
        let noise = std::array::from_fn(|i| std::array::from_fn(|j| noise[i][j]));
        let bias = [bias[0], bias[1], bias[2]];
        Ok(Self {
            position,
            orientation,
            misalignment,
            saturation_value,
            noise,
            bias,
            enabled: true,
        })
    }

    pub fn availability(&self) -> bool {
        self.enabled
    }
}

fn normalize(q: [f64; 4]) -> [f64; 4] {
    let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    q.map(|x| x / norm)
}
