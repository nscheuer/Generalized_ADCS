/// Native marker for the degree-6 zonal gravity model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct J6;

impl J6 {
    pub const DEGREE: u8 = 6;
}
