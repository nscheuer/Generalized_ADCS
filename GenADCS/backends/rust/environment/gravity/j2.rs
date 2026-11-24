/// Native marker for the degree-2 zonal gravity model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct J2;

impl J2 {
    pub const DEGREE: u8 = 2;
}
