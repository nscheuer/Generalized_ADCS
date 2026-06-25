__all__ = ["Constellation"]

import numpy as np
from tqdm import tqdm
from typing import List, Optional, Sequence

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.density_model import DensityModel
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants, ThirdBodyConstants
from ADCS.helpers.simresults import SimulationResults
from ADCS.formation.formation_world import FormationWorld
from ADCS.formation.satellite_agent import SatelliteAgent


class Constellation:
    r"""
    In-process orchestrator for multi-satellite (formation) simulation.

    Steps up to hundreds of :class:`~ADCS.formation.satellite_agent.SatelliteAgent`
    instances on one shared clock. Each timestep it:

    1. propagates every satellite's translational orbit one step (gravity model:
       central + J2, optional J3-J6 zonals and lunisolar third bodies),
    2. builds all satellites' environment at the new epoch in a single batched
       Skyfield/ppigrf pass (:meth:`Orbital_State.batch_at_epoch`),
    3. publishes every satellite's current truth state into a shared
       :class:`~ADCS.formation.formation_world.FormationWorld` so formation-aware
       goals can read neighbours, then
    4. steps each agent's GNC + attitude dynamics.

    The orbit is propagated separately from attitude here (gravity-only orbit is
    attitude-independent); attitude-coupled aerodynamic forces are layered on in
    a later phase via the agents.

    :param agents: Per-satellite :class:`SatelliteAgent` instances.
    :param os0_list: Initial :class:`Orbital_State` per agent (same epoch). Provides
        each satellite's initial ECI position/velocity and the shared start time.
    :param dt: Timestep [s].
    :param tf: Total duration [s].
    :param ephem: Shared ephemeris (defaults to ``os0_list[0].ephem``).
    :param density_model: Shared density model (defaults to the first state's,
        else a fresh :class:`DensityModel`).
    :param zonal_order: Highest zonal gravity degree for the orbit (2=J2 only..6).
    :param lunisolar: Enable Sun+Moon third-body perturbations on the orbit.
    :param world: Shared :class:`FormationWorld` (created if ``None``). When using
        formation-aware goals, pass the same world the goals reference.
    :param verbose: Show a progress bar.
    """

    def __init__(
        self,
        agents: Sequence[SatelliteAgent],
        os0_list: Sequence[Orbital_State],
        dt: float,
        tf: float,
        ephem=None,
        density_model: Optional[DensityModel] = None,
        zonal_order: int = 2,
        lunisolar: bool = False,
        aero: bool = False,
        world: Optional[FormationWorld] = None,
        verbose: bool = True,
    ) -> None:
        if len(agents) != len(os0_list):
            raise ValueError(f"agents ({len(agents)}) and os0_list ({len(os0_list)}) must have equal length")
        if len(agents) == 0:
            raise ValueError("Constellation requires at least one agent")

        self.agents = list(agents)
        self.os0_list = list(os0_list)
        self.dt = float(dt)
        self.tf = float(tf)
        self.zonal_order = int(zonal_order)
        self.lunisolar = bool(lunisolar)
        self.aero = bool(aero)
        self.verbose = bool(verbose)

        # Assign default ids where missing (used as run ids in the results).
        for i, ag in enumerate(self.agents):
            if ag.sat_id is None:
                ag.sat_id = i

        self.start_time = float(self.os0_list[0].J2000)
        if not all(np.isclose(float(os.J2000), self.start_time) for os in self.os0_list):
            raise ValueError("all initial orbital states must share the same epoch (J2000)")

        self.ephem = ephem if ephem is not None else self.os0_list[0].ephem
        if density_model is not None:
            self.density_model = density_model
        else:
            self.density_model = getattr(self.os0_list[0], "density_model", None) or DensityModel()

        self.world = world if world is not None else FormationWorld()

    def _higher_zonals(self):
        if self.zonal_order <= 2:
            return None
        return EarthConstants.Jcoeffs[1:self.zonal_order - 1]

    def _orbit_rk4_step(self, R_arr, V_arr, dt, higher_zonals, third_bodies, external_accels=None):
        r"""
        Advance every satellite's (R, V) by one RK4 gravity step (NumPy only).

        Uses :meth:`Orbital_State._orbit_dynamics_raw` per satellite; third bodies
        (Sun/Moon) and the optional per-satellite ``external_accels`` (e.g. the
        attitude-dependent aero drag+lift acceleration) are held fixed across the
        step (operator-split coupling). No Skyfield/ppigrf here.
        """
        mu = EarthConstants.mu_e
        Re = EarthConstants.R_e
        J2 = EarthConstants.J2coeff
        n = R_arr.shape[0]
        outR = np.empty_like(R_arr)
        outV = np.empty_like(V_arr)
        dyn = Orbital_State._orbit_dynamics_raw
        for i in range(n):
            r0 = R_arr[i]
            v0 = V_arr[i]
            ext = None if external_accels is None else external_accels[i]
            k1r, k1v = dyn(r0, v0, mu, Re, J2, True, higher_zonals, third_bodies, ext)
            k2r, k2v = dyn(r0 + 0.5 * dt * k1r, v0 + 0.5 * dt * k1v, mu, Re, J2, True, higher_zonals, third_bodies, ext)
            k3r, k3v = dyn(r0 + 0.5 * dt * k2r, v0 + 0.5 * dt * k2v, mu, Re, J2, True, higher_zonals, third_bodies, ext)
            k4r, k4v = dyn(r0 + dt * k3r, v0 + dt * k3v, mu, Re, J2, True, higher_zonals, third_bodies, ext)
            outR[i] = r0 + (dt / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
            outV[i] = v0 + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return outR, outV

    def _third_bodies(self, os_list):
        if not self.lunisolar:
            return None
        sun = np.asarray(os_list[0].S, dtype=float).reshape(3)   # ~identical for all sats
        moon = os_list[0].get_moon_eci()
        return [(ThirdBodyConstants.mu_sun, sun), (ThirdBodyConstants.mu_moon, moon)]

    def run(self) -> SimulationResults:
        r"""
        Run the constellation and return one :class:`RunResults` per satellite.

        :return: :class:`SimulationResults` with ``runs`` and ``run_ids`` aligned
            to the agent order.
        """
        N = int(self.tf / self.dt)
        sec2cent = TimeConstants.sec2cent
        cent2sec = TimeConstants.cent2sec
        dt = self.dt

        R_arr = np.vstack([np.asarray(os.R, dtype=float) for os in self.os0_list])
        V_arr = np.vstack([np.asarray(os.V, dtype=float) for os in self.os0_list])

        higher_zonals = self._higher_zonals()

        # Environment at the initial epoch (one batched pass).
        os_cur = Orbital_State.batch_at_epoch(R_arr, V_arr, self.start_time, self.ephem, self.density_model)

        for k in tqdm(range(N), desc="Simulating constellation", unit="step", disable=not self.verbose):
            t_k = self.start_time + k * dt * sec2cent
            t_kp1 = self.start_time + (k + 1) * dt * sec2cent

            # 1) advance orbits to the next epoch (gravity [+ optional aero])
            third_bodies = self._third_bodies(os_cur)
            external_accels = None
            if self.aero:
                # Attitude-dependent aero drag+lift from each satellite's current
                # attitude, held fixed across the step (operator-split coupling).
                external_accels = [ag.aero_accel_eci(os_cur[i]) for i, ag in enumerate(self.agents)]
            R_next, V_next = self._orbit_rk4_step(R_arr, V_arr, dt, higher_zonals, third_bodies, external_accels)
            # 2) batched environment at the next epoch
            os_next = Orbital_State.batch_at_epoch(R_next, V_next, t_kp1, self.ephem, self.density_model)

            # 3) publish synchronous truth snapshot for formation-aware goals
            for i, ag in enumerate(self.agents):
                self.world.update(ag.sat_id, R=R_arr[i], V=V_arr[i], q=ag.x[3:7], J2000=t_k)

            # 4) step each satellite's GNC + attitude dynamics
            for i, ag in enumerate(self.agents):
                ag.step(k, t_k, os_cur[i], os_next[i])

            R_arr, V_arr = R_next, V_next
            os_cur = os_next

        runs = [ag.results for ag in self.agents]
        run_ids = [ag.sat_id for ag in self.agents]
        return SimulationResults(runs=runs, run_ids=run_ids)
