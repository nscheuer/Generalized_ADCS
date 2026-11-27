pub const DESAT_NONE: i32 = 0;
pub const DESAT_HOGAN_SCHAUB: i32 = 1;

pub trait Desaturation: Send + Sync {
    fn kind(&self) -> i32;
    fn to_params(&self) -> Vec<f64>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoDesaturation;

impl Desaturation for NoDesaturation {
    fn kind(&self) -> i32 {
        DESAT_NONE
    }

    fn to_params(&self) -> Vec<f64> {
        vec![0.0, 0.0, 0.0]
    }
}
