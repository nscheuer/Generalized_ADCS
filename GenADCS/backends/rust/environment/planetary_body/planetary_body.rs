/// Native planetary constants used by Rust callers and Python adapters.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanetaryBody {
    name: String,
    gravitational_parameter: f64,
    reference_radius: f64,
}

impl PlanetaryBody {
    pub fn new(
        name: impl Into<String>,
        gravitational_parameter: f64,
        reference_radius: f64,
    ) -> Result<Self, String> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err("planetary body name must not be empty".into());
        }
        if !gravitational_parameter.is_finite() || gravitational_parameter <= 0.0 {
            return Err("gravitational_parameter must be finite and positive".into());
        }
        if !reference_radius.is_finite() || reference_radius <= 0.0 {
            return Err("reference_radius must be finite and positive".into());
        }
        Ok(Self {
            name,
            gravitational_parameter,
            reference_radius,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn gravitational_parameter(&self) -> f64 {
        self.gravitational_parameter
    }

    pub fn reference_radius(&self) -> f64 {
        self.reference_radius
    }

    pub fn set_name(&mut self, name: impl Into<String>) -> Result<(), String> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err("planetary body name must not be empty".into());
        }
        self.name = name;
        Ok(())
    }

    pub fn set_gravitational_parameter(&mut self, value: f64) -> Result<(), String> {
        if !value.is_finite() || value <= 0.0 {
            return Err("gravitational_parameter must be finite and positive".into());
        }
        self.gravitational_parameter = value;
        Ok(())
    }

    pub fn set_reference_radius(&mut self, value: f64) -> Result<(), String> {
        if !value.is_finite() || value <= 0.0 {
            return Err("reference_radius must be finite and positive".into());
        }
        self.reference_radius = value;
        Ok(())
    }
}
