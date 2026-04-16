"""
Basilisk ADCS simulation for a 3U CubeSat in an ISS orbit — nadir pointing.

Scenario matches Section V of the Generalized ADCS paper:
  3U CubeSat, 3 MTQ + 1 RW, nadir pointing, 1 orbit.

Spacecraft: 3U CubeSat (4 kg, 10x10x34 cm)
Actuators:  3 orthogonal magnetic torque bars (MTBs) + 1 reaction wheel (z-axis)
Sensors:    3-axis magnetometer (TAM) + 6 coarse sun sensors (CSS)
Orbit:      ISS-like (408 km altitude, 51.6 deg inclination)
Environment: Earth gravity (SPICE), WMM magnetic field, eclipse model

FSW chain:
  - hillPoint            → nadir-pointing reference attitude (Hill frame)
  - attTrackingError     → attitude / rate errors
  - mrpFeedback          → control torque Lr (uses RW speeds for gyroscopic comp.)
  - rwMotorTorque        → maps Lr z-component onto the single RW
  - torque2Dipole        → maps full Lr → MTB dipole request via cross-product law
  - dipoleMapping        → maps body dipole onto 3 MTB axes
  - MtbEffector          → applies MTB torques to dynamics

Author: Patrick McKeen / Claude
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend for WSL
import matplotlib.pyplot as plt

from Basilisk import __path__ as bskPath
from Basilisk.architecture import messaging
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion,
                                simIncludeGravBody, simIncludeRW,
                                unitTestSupport)
from Basilisk.utilities.supportDataTools.dataFetcher import get_path, DataFile

# ---------- simulation modules ----------
from Basilisk.simulation import (spacecraft, reactionWheelStateEffector,
                                 simpleNav, magneticFieldWMM, magnetometer,
                                 MtbEffector, coarseSunSensor, eclipse)

# ---------- FSW algorithm modules ----------
from Basilisk.fswAlgorithms import (hillPoint, attTrackingError, mrpFeedback,
                                    rwMotorTorque, tamComm,
                                    torque2Dipole, dipoleMapping)

# =====================================================================
#  Helper: 3U CubeSat physical parameters
# =====================================================================
MASS_SC = 4.0          # kg
# 3U body: 10 cm x 10 cm x 34 cm  (x, y, z)
LX, LY, LZ = 0.10, 0.10, 0.34   # metres

# Uniform-box MOI about CM   I_xx = m/12*(ly^2+lz^2), etc.
Ixx = MASS_SC / 12.0 * (LY**2 + LZ**2)
Iyy = MASS_SC / 12.0 * (LX**2 + LZ**2)
Izz = MASS_SC / 12.0 * (LX**2 + LY**2)

I_SC = [Ixx, 0., 0.,
        0., Iyy, 0.,
        0., 0., Izz]

# =====================================================================
#  MTB configuration: 3 orthogonal torque bars
# =====================================================================
NUM_MTB = 3
MAX_DIPOLE = 0.2   # A·m²  (typical small CubeSat MTQ)

# GtMatrix_B: each column is a unit torque-bar axis (row-major storage)
# bar 0 → x̂_B,  bar 1 → ŷ_B,  bar 2 → ẑ_B
GT_MATRIX_B = [1., 0., 0.,
               0., 1., 0.,
               0., 0., 1.]

# Pseudoinverse of Gt (for 3×3 identity it is identity)
STEERING_MATRIX = [1., 0., 0.,
                   0., 1., 0.,
                   0., 0., 1.]


def run(show_plots=True, sim_minutes=90.0):
    """
    Run the 3U CubeSat ADCS simulation.

    Args:
        show_plots: if True, display matplotlib figures at the end
        sim_minutes: simulation duration in minutes (default 90 = ~1 orbit)
    """

    # ------------------------------------------------------------------
    # 0.  Basilisk scaffolding
    # ------------------------------------------------------------------
    simTaskName = "simTask"
    simProcessName = "simProcess"
    scSim = SimulationBaseClass.SimBaseClass()

    simulationTime = macros.min2nano(sim_minutes)
    dt = macros.sec2nano(0.5)      # 0.5 s integration step

    dynProcess = scSim.CreateNewProcess(simProcessName)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, dt))

    # ------------------------------------------------------------------
    # 1.  Spacecraft hub
    # ------------------------------------------------------------------
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "CubeSat3U"
    scObject.hub.mHub = MASS_SC
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I_SC)
    scSim.AddModelToTask(simTaskName, scObject, ModelPriority=100)

    # ------------------------------------------------------------------
    # 2.  Gravity — Sun + Earth (SPICE for eclipse & CSS)
    # ------------------------------------------------------------------
    timeInitString = "2025 January 1, 00:00:0.0 (UTC)"
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravBodies = gravFactory.createBodies(['sun', 'earth'])
    gravBodies['earth'].isCentralBody = True
    mu = gravBodies['earth'].mu

    gravFactory.addBodiesTo(scObject)
    gravFactory.createSpiceInterface(time=timeInitString, epochInMsg=True)
    gravFactory.spiceObject.zeroBase = 'Earth'
    scSim.AddModelToTask(simTaskName, gravFactory.spiceObject, ModelPriority=99)

    # Indices into spiceObject.planetStateOutMsgs (order matches createBodies)
    sunIdx = 0
    earthIdx = 1
    epochMsg = gravFactory.epochMsg

    # ------------------------------------------------------------------
    # 3.  Magnetic field (WMM)
    # ------------------------------------------------------------------
    magModule = magneticFieldWMM.MagneticFieldWMM()
    magModule.ModelTag = "WMM"
    wmm_path = get_path(DataFile.MagneticFieldData.WMM)
    magModule.configureWMMFile(str(wmm_path))
    magModule.epochInMsg.subscribeTo(epochMsg)
    magModule.addSpacecraftToModel(scObject.scStateOutMsg)
    scSim.AddModelToTask(simTaskName, magModule, ModelPriority=98)

    # ------------------------------------------------------------------
    # 4.  Eclipse model
    # ------------------------------------------------------------------
    eclipseObj = eclipse.Eclipse()
    eclipseObj.ModelTag = "eclipse"
    eclipseObj.sunInMsg.subscribeTo(
        gravFactory.spiceObject.planetStateOutMsgs[sunIdx])
    eclipseObj.addPlanetToModel(
        gravFactory.spiceObject.planetStateOutMsgs[earthIdx])
    eclipseObj.addSpacecraftToModel(scObject.scStateOutMsg)
    scSim.AddModelToTask(simTaskName, eclipseObj, ModelPriority=97)

    # Sun SPICE message (for CSS)
    sunMsg = gravFactory.spiceObject.planetStateOutMsgs[sunIdx]

    # ------------------------------------------------------------------
    # 5.  Reaction wheel — single RW along ẑ_B
    # ------------------------------------------------------------------
    rwFactory = simIncludeRW.rwFactory()
    varRWModel = messaging.BalancedWheels

    gsHat_B = [0., 0., 1.]    # spin axis along body z
    RW0 = rwFactory.create('NanoAvionics_RW0', gsHat_B,
                           Omega_max=6000.,        # RPM
                           RWModel=varRWModel,
                           useRWfriction=False)

    numRW = rwFactory.getNumOfDevices()

    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwStateEffector.ModelTag = "RW_cluster"
    rwFactory.addToSpacecraft(scObject.ModelTag, rwStateEffector, scObject)
    scSim.AddModelToTask(simTaskName, rwStateEffector, ModelPriority=97)

    # ------------------------------------------------------------------
    # 6.  Magnetic torque bar effector (3 MTBs)
    # ------------------------------------------------------------------
    mtbEff = MtbEffector.MtbEffector()
    mtbEff.ModelTag = "MtbEff"
    scObject.addDynamicEffector(mtbEff)
    scSim.AddModelToTask(simTaskName, mtbEff, ModelPriority=96)

    # MTB configuration message
    mtbConfigParams = messaging.MTBArrayConfigMsgPayload()
    mtbConfigParams.numMTB = NUM_MTB
    mtbConfigParams.GtMatrix_B = GT_MATRIX_B
    mtbConfigParams.maxMtbDipoles = [MAX_DIPOLE] * NUM_MTB
    mtbParamsInMsg = messaging.MTBArrayConfigMsg().write(mtbConfigParams)

    # ------------------------------------------------------------------
    # 7.  Sensors — 3-axis magnetometer (TAM)
    # ------------------------------------------------------------------
    TAM = magnetometer.Magnetometer()
    TAM.ModelTag = "TAM_sensor"
    TAM.scaleFactor = 1.0
    TAM.senNoiseStd = [100e-9, 100e-9, 100e-9]   # 100 nT  noise (typical MEMS)
    scSim.AddModelToTask(simTaskName, TAM, ModelPriority=95)

    # TAM comm: rotates sensor-frame measurement into body frame
    tamCommObj = tamComm.tamComm()
    tamCommObj.dcm_BS = [1., 0., 0.,
                         0., 1., 0.,
                         0., 0., 1.]     # sensor aligned with body
    tamCommObj.ModelTag = "tamComm"
    scSim.AddModelToTask(simTaskName, tamCommObj, ModelPriority=94)

    # ------------------------------------------------------------------
    # 8.  Sensors — 6 coarse sun sensors (CSS) in ±x, ±y, ±z
    # ------------------------------------------------------------------
    cssConstellation = coarseSunSensor.CSSConstellation()
    cssConstellation.ModelTag = "CSSConstellation"

    # Normal vectors for 6 faces of a cube
    cssNormals = [[ 1., 0., 0.],
                  [-1., 0., 0.],
                  [ 0., 1., 0.],
                  [ 0.,-1., 0.],
                  [ 0., 0., 1.],
                  [ 0., 0.,-1.]]

    cssSensorList = []
    for i, nHat in enumerate(cssNormals):
        css = coarseSunSensor.CoarseSunSensor()
        css.ModelTag = f"CSS_{i}"
        css.fov = 80.0 * macros.D2R       # half-angle FOV
        css.scaleFactor = 1.0
        css.maxOutput = 1.0
        css.minOutput = 0.0
        css.senNoiseStd = 0.02             # ~2 % noise
        css.nHat_B = np.array(nHat)
        css.sunInMsg.subscribeTo(sunMsg)
        css.stateInMsg.subscribeTo(scObject.scStateOutMsg)
        css.sunEclipseInMsg.subscribeTo(eclipseObj.eclipseOutMsgs[0])
        cssSensorList.append(css)
        cssConstellation.appendCSS(css)

    scSim.AddModelToTask(simTaskName, cssConstellation, ModelPriority=93)

    # ------------------------------------------------------------------
    # 9.  Simple navigation (provides nav messages from truth)
    # ------------------------------------------------------------------
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(simTaskName, sNavObject, ModelPriority=92)

    # ------------------------------------------------------------------
    # 10.  FSW — Attitude guidance & control
    # ------------------------------------------------------------------

    # 10a.  Nadir (Hill-frame) pointing reference
    hillPointObj = hillPoint.hillPoint()
    hillPointObj.ModelTag = "hillPoint"
    scSim.AddModelToTask(simTaskName, hillPointObj, ModelPriority=80)

    # 10b.  Attitude tracking error
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    scSim.AddModelToTask(simTaskName, attError, ModelPriority=79)

    # 10c.  MRP feedback controller
    #       Gains must respect MTB torque authority.  Max MTB torque ≈ m_max × |B|
    #       ≈ 0.2 A·m² × 30 µT ≈ 6 µNm.  For Ixx ≈ 0.042 kg·m²:
    #         K  = 3e-4  → max proportional torque ≈ 3e-4 * 1.0 ≈ 0.3 mNm (clipped by MTB)
    #         P  = 5e-3  → max rate torque ≈ 5e-3 * 0.02 ≈ 0.1 mNm
    #       These are deliberately low so the MTBs un-saturate once the
    #       error shrinks, giving linear-regime behaviour for steady state.
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    mrpControl.K = 3e-4
    mrpControl.P = 5e-3
    mrpControl.Ki = -1           # negative → no integral term
    mrpControl.integralLimit = 0.0
    scSim.AddModelToTask(simTaskName, mrpControl, ModelPriority=78)

    # 10d.  torque2Dipole: converts the FULL mrpFeedback Lr → dipole request
    #       This gives the MTBs active attitude-control authority on all axes.
    #       The cross-product law inside torque2Dipole (τ = m × B) naturally
    #       projects the requested torque onto the achievable subspace ⊥ B.
    torque2DipoleObj = torque2Dipole.torque2Dipole()
    torque2DipoleObj.ModelTag = "torque2Dipole"
    scSim.AddModelToTask(simTaskName, torque2DipoleObj, ModelPriority=76)

    # 10e.  dipoleMapping: maps body dipole request → MTB dipole commands
    dipoleMappingObj = dipoleMapping.dipoleMapping()
    dipoleMappingObj.ModelTag = "dipoleMapping"
    dipoleMappingObj.steeringMatrix = STEERING_MATRIX
    scSim.AddModelToTask(simTaskName, dipoleMappingObj, ModelPriority=75)

    # 10f.  RW motor torque mapping
    #       With 1 RW on z-axis, specify a single control axis matching the wheel.
    #       mrpFeedback Lr z-component goes to the RW for fast z-axis control.
    rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
    rwMotorTorqueObj.ModelTag = "rwMotorTorque"
    rwMotorTorqueObj.controlAxes_B = [0., 0., 1.]
    scSim.AddModelToTask(simTaskName, rwMotorTorqueObj, ModelPriority=73)

    # ------------------------------------------------------------------
    # 11.  FSW configuration messages
    # ------------------------------------------------------------------
    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    vehicleConfigOut.ISCPntB_B = I_SC
    vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    fswRwParamMsg = rwFactory.getConfigMessage()

    # ------------------------------------------------------------------
    # 12.  Message connections  (subscribe inputs → outputs)
    # ------------------------------------------------------------------

    # Navigation
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

    # Hill-frame guidance: needs translational nav + planet ephemeris
    hillPointObj.transNavInMsg.subscribeTo(sNavObject.transOutMsg)
    # Earth is at the origin (zeroBase='Earth'), so zero'd ephemeris is correct
    celBodyData = messaging.EphemerisMsgPayload()
    celBodyInMsg = messaging.EphemerisMsg().write(celBodyData)
    hillPointObj.celBodyInMsg.subscribeTo(celBodyInMsg)

    # Guidance chain
    attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    attError.attRefInMsg.subscribeTo(hillPointObj.attRefOutMsg)

    # MRP feedback
    mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    mrpControl.vehConfigInMsg.subscribeTo(vcMsg)
    mrpControl.rwParamsInMsg.subscribeTo(fswRwParamMsg)
    mrpControl.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)

    # TAM sensor chain
    TAM.stateInMsg.subscribeTo(scObject.scStateOutMsg)
    TAM.magInMsg.subscribeTo(magModule.envOutMsgs[0])
    tamCommObj.tamInMsg.subscribeTo(TAM.tamDataOutMsg)

    # torque2Dipole — receives the FULL Lr from mrpFeedback for active MTB control
    torque2DipoleObj.tauRequestInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
    torque2DipoleObj.tamSensorBodyInMsg.subscribeTo(tamCommObj.tamOutMsg)

    # dipoleMapping
    dipoleMappingObj.dipoleRequestBodyInMsg.subscribeTo(
        torque2DipoleObj.dipoleRequestOutMsg)
    dipoleMappingObj.mtbArrayConfigParamsInMsg.subscribeTo(mtbParamsInMsg)

    # RW motor torque — receives Lr directly from mrpFeedback
    rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)
    rwMotorTorqueObj.vehControlInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)

    # Close the loop: RW effector receives motor commands
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(
        rwMotorTorqueObj.rwMotorTorqueOutMsg)

    # MTB effector receives dipole commands + mag field
    mtbEff.mtbCmdInMsg.subscribeTo(dipoleMappingObj.dipoleRequestMtbOutMsg)
    mtbEff.mtbParamsInMsg.subscribeTo(mtbParamsInMsg)
    mtbEff.magInMsg.subscribeTo(magModule.envOutMsgs[0])

    # ------------------------------------------------------------------
    # 13.  Initial conditions — ISS orbit
    # ------------------------------------------------------------------
    oe = orbitalMotion.ClassicElements()
    oe.a = (6371.0 + 408.0) * 1000.0    # ISS altitude ~408 km
    oe.e = 0.0005
    oe.i = 51.6 * macros.D2R            # ISS inclination
    oe.Omega = 30.0 * macros.D2R
    oe.omega = 0.0 * macros.D2R
    oe.f = 0.0 * macros.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN

    # Initial tumble — moderate rates typical after deployment
    scObject.hub.sigma_BNInit = [[0.3], [-0.2], [0.1]]
    scObject.hub.omega_BN_BInit = [[0.5 * macros.D2R],
                                   [-1.0 * macros.D2R],
                                   [2.0 * macros.D2R]]

    # ------------------------------------------------------------------
    # 14.  Data recorders
    # ------------------------------------------------------------------
    numDataPoints = 400
    samplingTime = unitTestSupport.samplingTime(
        simulationTime, dt, numDataPoints)

    # Attitude / rate errors
    attErrLog = attError.attGuidOutMsg.recorder(samplingTime)

    # RW speed
    rwSpeedLog = rwStateEffector.rwSpeedOutMsg.recorder(samplingTime)

    # RW motor torque
    rwMotorLog = rwMotorTorqueObj.rwMotorTorqueOutMsg.recorder(samplingTime)

    # Magnetic field (inertial)
    magFieldLog = magModule.envOutMsgs[0].recorder(samplingTime)

    # TAM measurement (body frame via tamComm)
    tamBodyLog = tamCommObj.tamOutMsg.recorder(samplingTime)

    # MTB dipole commands
    mtbCmdLog = dipoleMappingObj.dipoleRequestMtbOutMsg.recorder(samplingTime)

    # CSS constellation
    cssLog = cssConstellation.constellationOutMsg.recorder(samplingTime)

    # SC state (for orbit / attitude truth)
    scStateLog = scObject.scStateOutMsg.recorder(samplingTime)

    for rec in [attErrLog, rwSpeedLog, rwMotorLog, magFieldLog,
                tamBodyLog, mtbCmdLog, cssLog, scStateLog]:
        scSim.AddModelToTask(simTaskName, rec)

    # ------------------------------------------------------------------
    # 15.  Execute
    # ------------------------------------------------------------------
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simulationTime)
    print(f"Running {sim_minutes:.0f}-minute simulation …")
    scSim.ExecuteSimulation()
    print("Simulation complete.")

    # ------------------------------------------------------------------
    # 16.  Retrieve logged data
    # ------------------------------------------------------------------
    timeMin = attErrLog.times() * macros.NANO2MIN

    sigmaBR   = attErrLog.sigma_BR
    omegaBR   = attErrLog.omega_BR_B
    rwSpeeds  = rwSpeedLog.wheelSpeeds
    rwTorque  = rwMotorLog.motorTorque
    magField  = magFieldLog.magField_N
    tamBody   = tamBodyLog.tam_B
    mtbDipole = mtbCmdLog.mtbDipoleCmds
    cssData   = cssLog.CosValue

    # ------------------------------------------------------------------
    # 17.  Plots
    # ------------------------------------------------------------------
    figDir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(figDir, exist_ok=True)

    plt.close("all")

    # --- Attitude error ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(3):
        ax.plot(timeMin, sigmaBR[:, i], label=rf"$\sigma_{{BR,{i}}}$")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel(r"Attitude Error $\sigma_{B/R}$")
    ax.set_title("MRP Attitude Error")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(figDir, "bsk_att_error.png"), dpi=150)

    # --- Angular rate error ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(3):
        ax.plot(timeMin, omegaBR[:, i] * macros.R2D,
                label=rf"$\omega_{{BR,{i}}}$")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Rate Error [deg/s]")
    ax.set_title("Body Rate Tracking Error")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(figDir, "bsk_rate_error.png"), dpi=150)

    # --- RW speed ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timeMin, rwSpeeds[:, 0] / macros.RPM, label="RW0 (z-axis)")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("RW Speed [RPM]")
    ax.set_title("Reaction Wheel Speed")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(figDir, "bsk_rw_speed.png"), dpi=150)

    # --- RW motor torque ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timeMin, rwTorque[:, 0] * 1e3, label="RW0 torque")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Motor Torque [mNm]")
    ax.set_title("RW Motor Torque")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(figDir, "bsk_rw_torque.png"), dpi=150)

    # --- Magnetic field (inertial) ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(3):
        ax.plot(timeMin, magField[:, i] * 1e6,
                label=rf"$B_{{N,{i}}}$")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Magnetic Field [µT]")
    ax.set_title("Earth Magnetic Field (Inertial Frame)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(figDir, "bsk_mag_field.png"), dpi=150)

    # --- TAM body measurement ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(3):
        ax.plot(timeMin, tamBody[:, i] * 1e6,
                label=rf"$B_{{B,{i}}}$")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("TAM Reading [µT]")
    ax.set_title("Magnetometer (Body Frame)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(figDir, "bsk_tam_body.png"), dpi=150)

    # --- MTB dipole commands ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(NUM_MTB):
        ax.plot(timeMin, mtbDipole[:, i] * 1e3,
                label=f"MTB {i}")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Dipole Command [mA·m²]")
    ax.set_title("Magnetic Torque Bar Dipole Commands")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(figDir, "bsk_mtb_dipoles.png"), dpi=150)

    # --- CSS signals ---
    fig, ax = plt.subplots(figsize=(10, 4))
    cssLabels = ["+X", "−X", "+Y", "−Y", "+Z", "−Z"]
    for i in range(min(6, cssData.shape[1])):
        ax.plot(timeMin, cssData[:, i], label=cssLabels[i], alpha=0.8)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("CSS Output [normalised]")
    ax.set_title("Coarse Sun Sensor Readings")
    ax.legend(ncol=3)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(figDir, "bsk_css_signals.png"), dpi=150)

    print(f"\nPlots saved to {figDir}/bsk_*.png")

    if show_plots:
        plt.show()

    plt.close("all")


# =====================================================================
if __name__ == "__main__":
    run(show_plots=False, sim_minutes=90.0)
