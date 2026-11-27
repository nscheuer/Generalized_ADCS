/// Native collection of actuator, sensor, and estimator components.
#[derive(Clone, Debug, PartialEq)]
pub struct ADCSSubsystem<W = (), M = (), S = (), E = (), SM = (), C = ()> {
    pub reaction_wheels: Vec<W>,
    pub magnetorquers: Vec<M>,
    pub sensors: Vec<S>,
    #[allow(non_snake_case)]
    pub N: usize,
    pub estimator: Option<E>,
    pub state_machine: Option<SM>,
    pub controller: Option<C>,
}

impl<W, M, S, E, SM, C> Default for ADCSSubsystem<W, M, S, E, SM, C> {
    fn default() -> Self {
        Self {
            reaction_wheels: Vec::new(),
            magnetorquers: Vec::new(),
            sensors: Vec::new(),
            N: 1,
            estimator: None,
            state_machine: None,
            controller: None,
        }
    }
}

impl<W, M, S, E> ADCSSubsystem<W, M, S, E, (), ()> {
    pub fn with_components(
        reaction_wheels: Vec<W>,
        magnetorquers: Vec<M>,
        sensors: Vec<S>,
        n: usize,
        estimator: Option<E>,
    ) -> Result<Self, &'static str> {
        Self::with_full_components(reaction_wheels, magnetorquers, sensors, n, estimator, None, None)
    }
}

impl<W, M, S, E, SM, C> ADCSSubsystem<W, M, S, E, SM, C> {
    pub fn new(reaction_wheels: Vec<W>, magnetorquers: Vec<M>) -> Self {
        Self {
            reaction_wheels,
            magnetorquers,
            sensors: Vec::new(),
            N: 1,
            estimator: None,
            state_machine: None,
            controller: None,
        }
    }

    pub fn with_full_components(
        reaction_wheels: Vec<W>,
        magnetorquers: Vec<M>,
        sensors: Vec<S>,
        n: usize,
        estimator: Option<E>,
        state_machine: Option<SM>,
        controller: Option<C>,
    ) -> Result<Self, &'static str> {
        if n == 0 {
            return Err("N must be a positive integer");
        }
        Ok(Self {
            reaction_wheels,
            magnetorquers,
            sensors,
            N: n,
            estimator,
            state_machine,
            controller,
        })
    }
}
