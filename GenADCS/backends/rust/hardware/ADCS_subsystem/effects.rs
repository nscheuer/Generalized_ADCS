/// Native interface for an effect that transforms a value over a time step.
pub trait Effect<T> {
    type Error;

    fn apply(&self, value: T, dt: f64) -> Result<T, Self::Error>;
}

/// Apply native effects in order, stopping at the first error.
pub fn apply_sequence<T, E>(mut value: T, effects: &[E], dt: f64) -> Result<T, E::Error>
where
    E: Effect<T>,
{
    for effect in effects {
        value = effect.apply(value, dt)?;
    }
    Ok(value)
}
