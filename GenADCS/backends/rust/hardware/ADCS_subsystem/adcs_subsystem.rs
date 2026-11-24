/// Native collection of actuator and subsystem components.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ADCSSubsystem<W = (), M = ()> {
    pub reaction_wheels: Vec<W>,
    pub magnetorquers: Vec<M>,
}

impl<W, M> ADCSSubsystem<W, M> {
    pub fn new(reaction_wheels: Vec<W>, magnetorquers: Vec<M>) -> Self {
        Self {
            reaction_wheels,
            magnetorquers,
        }
    }
}
