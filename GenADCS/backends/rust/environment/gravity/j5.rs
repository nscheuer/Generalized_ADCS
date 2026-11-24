/// Native marker for the degree-5 zonal gravity model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct J5;

impl J5 {
    pub const DEGREE: u8 = 5;
}
