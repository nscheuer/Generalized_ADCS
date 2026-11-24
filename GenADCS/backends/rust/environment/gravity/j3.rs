/// Native marker for the degree-3 zonal gravity model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct J3;

impl J3 {
    pub const DEGREE: u8 = 3;
}
