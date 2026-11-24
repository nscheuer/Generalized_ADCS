//! Native implementation of the bounded random-walk bias model.
//!
//! Values are stored in row-major flattened form. `shape` records the broadcast
//! dimensions for callers that need to preserve them.

const CENTURY_SECONDS: f64 = 100.0 * 365.25 * 24.0 * 3600.0;
const BIAS_STREAM: u32 = 0x8000_0000;
const DEFAULT_SEED: u64 = 0;

#[derive(Clone, Debug, PartialEq)]
pub struct Bias {
    pub bias: Vec<f64>,
    pub std_bias: Vec<f64>,
    pub estimated_std_bias: Vec<f64>,
    pub last_bias_time: f64,
    pub lower_bound: Vec<f64>,
    pub upper_bound: Vec<f64>,
    pub shape: Vec<usize>,
    pub random_tick: u64,
}

impl Bias {
    /// Construct a bias from already broadcast, equally sized flattened data.
    pub fn new(
        bias: Vec<f64>,
        std_bias: Vec<f64>,
        lower_bound: Vec<f64>,
        upper_bound: Vec<f64>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        let size = bias.len();
        if size == 0
            || std_bias.len() != size
            || lower_bound.len() != size
            || upper_bound.len() != size
            || shape.iter().product::<usize>() != size
        {
            return Err("bias arrays must have the same nonzero size and match shape".into());
        }
        if lower_bound
            .iter()
            .zip(&upper_bound)
            .any(|(lower, upper)| lower > upper)
        {
            return Err("Each lower bound must be <= the corresponding upper bound.".into());
        }
        Ok(Self {
            estimated_std_bias: vec![0.0; size],
            bias,
            std_bias,
            last_bias_time: f64::NAN,
            lower_bound,
            upper_bound,
            shape,
            random_tick: 0,
        })
    }

    pub fn scalar(bias: f64, std_bias: f64) -> Self {
        Self::new(
            vec![bias],
            vec![std_bias],
            vec![f64::NEG_INFINITY],
            vec![f64::INFINITY],
            vec![1],
        )
        .expect("scalar bias has valid dimensions")
    }

    pub fn is_active(&self) -> bool {
        self.bias.iter().any(|value| *value != 0.0)
            || self.std_bias.iter().any(|value| *value != 0.0)
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }

    pub fn broadcast_to(&self, length: usize, owner: &str) -> Result<Self, String> {
        if self.std_bias.len() == length {
            return Ok(self.clone());
        }
        if self.std_bias.len() != 1 {
            return Err(format!(
                "{owner} bias model has {} entries for a {length}-output sensor",
                self.std_bias.len()
            ));
        }
        let repeat = |values: &[f64]| {
            if values.len() == 1 {
                vec![values[0]; length]
            } else {
                values.to_vec()
            }
        };
        Self::new(
            repeat(&self.bias),
            repeat(&self.std_bias),
            repeat(&self.lower_bound),
            repeat(&self.upper_bound),
            vec![length],
        )
    }

    pub fn process_psd(&self) -> Vec<Vec<f64>> {
        let n = self.std_bias.len();
        let mut result = vec![vec![0.0; n]; n];
        for (i, std) in self.std_bias.iter().enumerate() {
            result[i][i] = std * std;
        }
        result
    }

    pub fn process_covariance(&self, dt: f64) -> Result<Vec<Vec<f64>>, String> {
        if !dt.is_finite() || dt < 0.0 {
            return Err("dt must be finite and non-negative".into());
        }
        Ok(self
            .process_psd()
            .into_iter()
            .map(|row| row.into_iter().map(|value| value * dt).collect())
            .collect())
    }

    /// Advance with the native counter-based random stream.
    pub fn update_bias(&mut self, j2000: f64, seed: u64, stream: u32) -> Result<(), String> {
        if !j2000.is_finite() {
            return Err("j2000 must be finite".into());
        }
        if self.last_bias_time.is_nan() {
            self.last_bias_time = j2000;
            return Ok(());
        }
        let elapsed_centuries = j2000 - self.last_bias_time;
        if elapsed_centuries <= 0.0 {
            return Ok(());
        }
        let normals = philox_native::philox::standard_normals(
            seed,
            self.random_tick,
            stream,
            self.bias.len(),
        );
        self.update_with_normals(j2000, elapsed_centuries, &normals, true)
    }

    /// Advance using variates supplied by an external random source.
    pub fn update_bias_with_normals(
        &mut self,
        j2000: f64,
        standard_normals: &[f64],
    ) -> Result<(), String> {
        if !j2000.is_finite() {
            return Err("j2000 must be finite".into());
        }
        if self.last_bias_time.is_nan() {
            self.last_bias_time = j2000;
            return Ok(());
        }
        let elapsed_centuries = j2000 - self.last_bias_time;
        if elapsed_centuries <= 0.0 {
            return Ok(());
        }
        self.update_with_normals(j2000, elapsed_centuries, standard_normals, false)
    }

    fn update_with_normals(
        &mut self,
        j2000: f64,
        elapsed_centuries: f64,
        standard_normals: &[f64],
        advance_random_tick: bool,
    ) -> Result<(), String> {
        if standard_normals.len() != self.bias.len() {
            return Err("one standard-normal variate is required per bias component".into());
        }
        let scale = (elapsed_centuries * CENTURY_SECONDS).sqrt();
        for i in 0..self.bias.len() {
            let candidate = self.bias[i] + self.std_bias[i] * scale * standard_normals[i];
            self.bias[i] = candidate.clamp(self.lower_bound[i], self.upper_bound[i]);
        }
        self.last_bias_time = j2000;
        if advance_random_tick {
            self.random_tick = self.random_tick.wrapping_add(1);
        }
        Ok(())
    }

    /// Advance using the default seed and bias-specific stream.
    pub fn update_bias_default(&mut self, j2000: f64) -> Result<(), String> {
        self.update_bias(j2000, DEFAULT_SEED, BIAS_STREAM)
    }

    pub fn get_bias(&self, _j2000: f64) -> &[f64] {
        &self.bias
    }

    pub fn apply(&self, value: &[f64]) -> Result<Vec<f64>, String> {
        if value.len() != self.bias.len() {
            return Err("value and bias must have matching shapes".into());
        }
        Ok(value
            .iter()
            .zip(&self.bias)
            .map(|(value, bias)| value + bias)
            .collect())
    }
}
