pub const FF_PASSTHROUGH: i32 = 0;

pub trait Feedforward: Send + Sync {
    fn kind(&self) -> i32;
    fn to_params(&self) -> Vec<f64>;
}
