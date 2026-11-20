import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ADCS_Omni import (
    ADCSSubsystem,
    AttitudeState,
    CHAOS8,
    CoupledPropagator,
    CoupledSolver,
    DE421,
    DynamicsConfig,
    Earth,
    Environment,
    FixedBias,
    FlickerNoise,
    GaussMarkovNoise,
    GaussianWhiteNoise,
    InternalState,
    JGravityModel,
    LinearDrift,
    Magnetorquer,
    NRLMSISE00,
    OrbitalState,
    QuantizationNoise,
    RandomWalkBias,
    ReactionWheel,
    Satellite,
    SatelliteGeometry,
    SatellitePanel,
    ScaleFactorError,
    SimulationConfig,
    State,
    StateMachine,
    Propagator,
    Solver,
    simulate,
)


def build_6u_simulation():
    mass = 12.0
    geometry = SatelliteGeometry(
        mass=mass,
        inertia=np.diag((0.13, 0.10, 0.05)),
        center_of_mass=(0.0, 0.0, 0.0),
        panels=[
            SatellitePanel((0.0, 0.0, 0.15), (0.0, 0.0, 0.0), (0.10, 0.20), 0.20, 0.70, 0.10, 2.2),
            SatellitePanel((0.0, 0.0, -0.15), (np.pi, 0.0, 0.0), (0.10, 0.20), 0.20, 0.70, 0.10, 2.2),
            SatellitePanel((0.05, 0.0, 0.0), (0.0, np.pi / 2.0, 0.0), (0.20, 0.30), 0.20, 0.70, 0.10, 2.2),
            SatellitePanel((-0.05, 0.0, 0.0), (0.0, -np.pi / 2.0, 0.0), (0.20, 0.30), 0.20, 0.70, 0.10, 2.2),
            SatellitePanel((0.0, 0.10, 0.0), (-np.pi / 2.0, 0.0, 0.0), (0.10, 0.30), 0.20, 0.70, 0.10, 2.2),
            SatellitePanel((0.0, -0.10, 0.0), (np.pi / 2.0, 0.0, 0.0), (0.10, 0.30), 0.20, 0.70, 0.10, 2.2),
        ],
    )

    wheel_axes = (
        (1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    )
    wheels = [
        ReactionWheel(
            position=position,
            orientation=axis,
            misalignment=(0.05, -0.03, 0.02),
            rotational_inertia=2.5e-5,
            max_torque=2.0e-3,
            max_momentum=2.0e-2,
            effects=[
                FixedBias(2.0e-6),
                GaussMarkovNoise(5.0e-7, 100.0),
                RandomWalkBias(1.0e-8),
                QuantizationNoise(1.0e-7),
                ScaleFactorError(2.0e-3),
            ],
        )
        for position, axis in zip(
            ((0.035, 0.075, 0.10), (-0.035, 0.075, 0.10), (0.035, -0.075, -0.10), (-0.035, -0.075, -0.10)),
            wheel_axes,
        )
    ]

    magnetorquers = [
        Magnetorquer(
            position=position,
            orientation=axis,
            misalignment=(0.1, -0.08, 0.04),
            max_dipole=0.2,
            effects=[
                FixedBias(1.0e-3),
                GaussianWhiteNoise(2.0e-4),
                LinearDrift(1.0e-6),
                FlickerNoise(5.0e-5),
                ScaleFactorError(3.0e-3),
            ],
        )
        for position, axis in zip(
            ((0.04, 0.0, 0.0), (0.0, 0.08, 0.0), (0.0, 0.0, 0.12)),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
    ]

    satellite = Satellite(
        geometry=geometry,
        adcs=ADCSSubsystem(reaction_wheels=wheels, magnetorquers=magnetorquers),
        state_machine=StateMachine(initial_mode="detumble"),
    )

    environment = Environment(
        planetary_body=Earth(),
        atmosphere=NRLMSISE00(),
        magnetic_field=CHAOS8(),
        j_gravity=JGravityModel(degrees=(2, 3, 4, 5, 6)),
        ephemeris=DE421(),
    )

    initial_state = State(
        attitude=AttitudeState(
            orientation=(1.0, 0.0, 0.0, 0.0),
            rates=(0.02, -0.015, 0.01),
        ),
        orbit=OrbitalState.from_cartesian(
            position=(6_878_137.0, 0.0, 0.0),
            velocity=(0.0, 7_612.6, 0.0),
            julian_date=2_461_000.5,
            ephemeris=environment.ephemeris,
        ),
        internal=InternalState(wheel_momentum=(0.0, 0.0, 0.0, 0.0)),
    )

    dynamics = DynamicsConfig(
        drag_torque=True,
        drag_torque_model="panel_facets",
        srp_torque=True,
        srp_torque_model="panel_facets",
        residual_dipole_torque=True,
        residual_dipole_model=(0.02, -0.01, 0.015),
        gravity_gradient_torque=True,
        gravity_gradient_model="full_inertia_tensor",
        drag_force=True,
        drag_force_model="panel_facets",
        srp_force=True,
        srp_force_model="panel_facets",
        j2_force=True,
        j3_force=True,
        j4_force=True,
        j5_force=True,
        j6_force=True,
        third_body_forces=True,
        third_body_model=("sun", "moon"),
    )

    dt = 0.25
    coupled_propagator = CoupledPropagator(dt=dt)
    propagator = Propagator(coupled=coupled_propagator)
    solver = Solver(
        dynamics=dynamics,
        coupled=CoupledSolver(propagator=coupled_propagator),
    )

    config = SimulationConfig(
        satellite=satellite,
        environment=environment,
        state=initial_state,
        propagator=propagator,
        solver=solver,
        backend="Python",
    )
    return config


def main():
    config = build_6u_simulation()
    try:
        result = simulate(config, tf=600.0)
        print(result)
    except NotImplementedError as error:
        print("Configuration is complete; the dynamics solvers are stubs.")
        print(f"simulate(config, tf=600.0) reached the simulation stub: {error}")


if __name__ == "__main__":
    main()
