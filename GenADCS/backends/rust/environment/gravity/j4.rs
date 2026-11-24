/// Native marker for the degree-4 zonal gravity model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct J4;

impl J4 {
    pub const DEGREE: u8 = 4;
}
