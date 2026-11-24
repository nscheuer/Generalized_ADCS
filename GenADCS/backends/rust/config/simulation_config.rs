/// Backend selection for simulation execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimulationBackend {
    Python,
    C,
    Rust,
    Jax,
}

impl SimulationBackend {
    pub fn parse(value: &str) -> Result<Self, &'static str> {
        match value {
            "Python" => Ok(Self::Python),
            "C" => Ok(Self::C),
            "Rust" => Ok(Self::Rust),
            "JAX" => Ok(Self::Jax),
            _ => Err("backend must be Python, C, Rust, or JAX"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Python => "Python",
            Self::C => "C",
            Self::Rust => "Rust",
            Self::Jax => "JAX",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimulationConfig {
    pub backend: String,
    pub julian_date: f64,
}

impl SimulationConfig {
    pub fn new(backend: &str, julian_date: f64) -> Result<Self, &'static str> {
        Ok(Self {
            backend: SimulationBackend::parse(backend)?.as_str().to_owned(),
            julian_date,
        })
    }
}
