/// Native collection of actuator, sensor, and estimator components.
#[derive(Clone, Debug, PartialEq)]
pub struct ADCSSubsystem<W = (), M = (), S = (), E = ()> {
    pub reaction_wheels: Vec<W>,
    pub magnetorquers: Vec<M>,
    pub sensors: Vec<S>,
    #[allow(non_snake_case)]
    pub N: usize,
    pub estimator: Option<E>,
}

impl<W, M, S, E> Default for ADCSSubsystem<W, M, S, E> {
    fn default() -> Self {
        Self { reaction_wheels: Vec::new(), magnetorquers: Vec::new(), sensors: Vec::new(), N: 1, estimator: None }
    }
}

impl<W, M, S, E> ADCSSubsystem<W, M, S, E> {
    pub fn new(reaction_wheels: Vec<W>, magnetorquers: Vec<M>) -> Self {
        Self {
            reaction_wheels,
            magnetorquers,
            sensors: Vec::new(),
            N: 1,
            estimator: None,
        }
    }

    pub fn with_components(
        reaction_wheels: Vec<W>,
        magnetorquers: Vec<M>,
        sensors: Vec<S>,
        n: usize,
        estimator: Option<E>,
    ) -> Result<Self, &'static str> {
        if n == 0 {
            return Err("N must be a positive integer");
        }
        Ok(Self { reaction_wheels, magnetorquers, sensors, N: n, estimator })
    }
}
