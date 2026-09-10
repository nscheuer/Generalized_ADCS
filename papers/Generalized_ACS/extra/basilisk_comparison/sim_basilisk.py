"""
Basilisk — 3U CubeSat nadir-pointing simulation.
Scenario: 3U CubeSat, 3 MTQ + 1 RW, nadir pointing, ISS orbit, 1 orbit.
"""
import os, time
import numpy as np
from Basilisk import __path__ as bskPath
from Basilisk.architecture import messaging
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion,
                                simIncludeGravBody, simIncludeRW,
                                unitTestSupport)
from Basilisk.utilities.supportDataTools.dataFetcher import get_path, DataFile
from Basilisk.simulation import (spacecraft, reactionWheelStateEffector,
                                 simpleNav, magneticFieldWMM, magnetometer,
                                 MtbEffector, coarseSunSensor, eclipse)
from Basilisk.fswAlgorithms import (hillPoint, attTrackingError, mrpFeedback,
                                    rwMotorTorque, tamComm,
                                    torque2Dipole, dipoleMapping)

# ── Physical parameters ───────────────────────────────────────────────
MASS = 4.0
LX, LY, LZ = 0.10, 0.10, 0.34
I_SC = [MASS/12*(LY**2+LZ**2), 0., 0.,
        0., MASS/12*(LX**2+LZ**2), 0.,
        0., 0., MASS/12*(LX**2+LY**2)]
NUM_MTB = 3
MAX_DIPOLE = 0.2

# ── Basilisk scaffolding ──────────────────────────────────────────────
simTaskName, simProcessName = "simTask", "simProcess"
scSim = SimulationBaseClass.SimBaseClass()
SIM_TIME = macros.min2nano(90.)
DT = macros.sec2nano(1.0)
dynProcess = scSim.CreateNewProcess(simProcessName)
dynProcess.addTask(scSim.CreateNewTask(simTaskName, DT))

# ── 1. Spacecraft hub ─────────────────────────────────────────────────
scObject = spacecraft.Spacecraft()
scObject.ModelTag = "CubeSat3U"
scObject.hub.mHub = MASS
scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I_SC)
scSim.AddModelToTask(simTaskName, scObject, 100)

# ── 2. Gravity (Sun + Earth via SPICE) ────────────────────────────────
gravFactory = simIncludeGravBody.gravBodyFactory()
gravBodies = gravFactory.createBodies(['sun', 'earth'])
gravBodies['earth'].isCentralBody = True
mu = gravBodies['earth'].mu
gravFactory.addBodiesTo(scObject)
gravFactory.createSpiceInterface(time="2025 January 1, 00:00:0.0 (UTC)",
                                 epochInMsg=True)
gravFactory.spiceObject.zeroBase = 'Earth'
scSim.AddModelToTask(simTaskName, gravFactory.spiceObject, 99)
sunIdx, earthIdx = 0, 1
epochMsg = gravFactory.epochMsg

# ── 3. Magnetic field (WMM) ───────────────────────────────────────────
magModule = magneticFieldWMM.MagneticFieldWMM()
magModule.ModelTag = "WMM"
magModule.configureWMMFile(str(get_path(DataFile.MagneticFieldData.WMM)))
magModule.epochInMsg.subscribeTo(epochMsg)
magModule.addSpacecraftToModel(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, magModule, 98)

# ── 4. Eclipse ────────────────────────────────────────────────────────
eclipseObj = eclipse.Eclipse()
eclipseObj.ModelTag = "eclipse"
eclipseObj.sunInMsg.subscribeTo(
    gravFactory.spiceObject.planetStateOutMsgs[sunIdx])
eclipseObj.addPlanetToModel(
    gravFactory.spiceObject.planetStateOutMsgs[earthIdx])
eclipseObj.addSpacecraftToModel(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, eclipseObj, 97)
sunMsg = gravFactory.spiceObject.planetStateOutMsgs[sunIdx]

# ── 5. Reaction wheel (z-axis) ────────────────────────────────────────
rwFactory = simIncludeRW.rwFactory()
RW0 = rwFactory.create('NanoAvionics_RW0', [0., 0., 1.],
                        Omega_max=6000.,
                        RWModel=messaging.BalancedWheels,
                        useRWfriction=False)
rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
rwStateEffector.ModelTag = "RW_cluster"
rwFactory.addToSpacecraft(scObject.ModelTag, rwStateEffector, scObject)
scSim.AddModelToTask(simTaskName, rwStateEffector, 96)

# ── 6. Magnetic torque bars (3 orthogonal) ────────────────────────────
mtbEff = MtbEffector.MtbEffector()
mtbEff.ModelTag = "MtbEff"
scObject.addDynamicEffector(mtbEff)
scSim.AddModelToTask(simTaskName, mtbEff, 95)

mtbConfigParams = messaging.MTBArrayConfigMsgPayload()
mtbConfigParams.numMTB = NUM_MTB
mtbConfigParams.GtMatrix_B = [1.,0.,0., 0.,1.,0., 0.,0.,1.]
mtbConfigParams.maxMtbDipoles = [MAX_DIPOLE] * NUM_MTB
mtbParamsInMsg = messaging.MTBArrayConfigMsg().write(mtbConfigParams)

# ── 7. Sensors — magnetometer ─────────────────────────────────────────
TAM = magnetometer.Magnetometer()
TAM.ModelTag = "TAM_sensor"
TAM.scaleFactor = 1.0
TAM.senNoiseStd = [100e-9, 100e-9, 100e-9]
scSim.AddModelToTask(simTaskName, TAM, 94)

tamCommObj = tamComm.tamComm()
tamCommObj.dcm_BS = [1.,0.,0., 0.,1.,0., 0.,0.,1.]
tamCommObj.ModelTag = "tamComm"
scSim.AddModelToTask(simTaskName, tamCommObj, 93)

# ── 8. Sensors — 6 coarse sun sensors ─────────────────────────────────
cssConstellation = coarseSunSensor.CSSConstellation()
cssConstellation.ModelTag = "CSSConstellation"
cssList = []   # prevent garbage collection of sensor objects
for i, nHat in enumerate([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]):
    css = coarseSunSensor.CoarseSunSensor()
    css.ModelTag = f"CSS_{i}"
    css.fov = 80.0 * macros.D2R
    css.scaleFactor = 1.0
    css.maxOutput = 1.0
    css.minOutput = 0.0
    css.senNoiseStd = 0.02
    css.nHat_B = np.array(nHat, dtype=float)
    css.sunInMsg.subscribeTo(sunMsg)
    css.stateInMsg.subscribeTo(scObject.scStateOutMsg)
    css.sunEclipseInMsg.subscribeTo(eclipseObj.eclipseOutMsgs[0])
    cssList.append(css)
    cssConstellation.appendCSS(css)
scSim.AddModelToTask(simTaskName, cssConstellation, 92)

# ── 9. Navigation ─────────────────────────────────────────────────────
sNavObject = simpleNav.SimpleNav()
sNavObject.ModelTag = "SimpleNavigation"
scSim.AddModelToTask(simTaskName, sNavObject, 91)

# ── 10. FSW — Hill-frame nadir guidance ───────────────────────────────
hillPointObj = hillPoint.hillPoint()
hillPointObj.ModelTag = "hillPoint"
scSim.AddModelToTask(simTaskName, hillPointObj, 80)

# ── 11. FSW — Attitude error ──────────────────────────────────────────
attError = attTrackingError.attTrackingError()
attError.ModelTag = "attError"
scSim.AddModelToTask(simTaskName, attError, 79)

# ── 12. FSW — MRP feedback controller ─────────────────────────────────
mrpControl = mrpFeedback.mrpFeedback()
mrpControl.ModelTag = "mrpFeedback"
mrpControl.K = 3e-4
mrpControl.P = 5e-3
mrpControl.Ki = -1
mrpControl.integralLimit = 0.0
scSim.AddModelToTask(simTaskName, mrpControl, 78)

# ── 13. FSW — torque2Dipole (MTB active attitude control) ─────────────
torque2DipoleObj = torque2Dipole.torque2Dipole()
torque2DipoleObj.ModelTag = "torque2Dipole"
scSim.AddModelToTask(simTaskName, torque2DipoleObj, 76)

# ── 14. FSW — dipoleMapping ───────────────────────────────────────────
dipoleMappingObj = dipoleMapping.dipoleMapping()
dipoleMappingObj.ModelTag = "dipoleMapping"
dipoleMappingObj.steeringMatrix = [1.,0.,0., 0.,1.,0., 0.,0.,1.]
scSim.AddModelToTask(simTaskName, dipoleMappingObj, 75)

# ── 15. FSW — RW motor torque (z-axis only) ───────────────────────────
rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
rwMotorTorqueObj.ModelTag = "rwMotorTorque"
rwMotorTorqueObj.controlAxes_B = [0., 0., 1.]
scSim.AddModelToTask(simTaskName, rwMotorTorqueObj, 73)

# ── 16. Configuration messages ────────────────────────────────────────
vcMsg = messaging.VehicleConfigMsg().write(
    messaging.VehicleConfigMsgPayload(ISCPntB_B=I_SC))
fswRwParamMsg = rwFactory.getConfigMessage()

# ── 17. Message wiring (12 connections) ───────────────────────────────
sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

hillPointObj.transNavInMsg.subscribeTo(sNavObject.transOutMsg)
celBodyData = messaging.EphemerisMsgPayload()
hillPointObj.celBodyInMsg.subscribeTo(
    messaging.EphemerisMsg().write(celBodyData))

attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
attError.attRefInMsg.subscribeTo(hillPointObj.attRefOutMsg)

mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
mrpControl.vehConfigInMsg.subscribeTo(vcMsg)
mrpControl.rwParamsInMsg.subscribeTo(fswRwParamMsg)
mrpControl.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)

TAM.stateInMsg.subscribeTo(scObject.scStateOutMsg)
TAM.magInMsg.subscribeTo(magModule.envOutMsgs[0])
tamCommObj.tamInMsg.subscribeTo(TAM.tamDataOutMsg)

torque2DipoleObj.tauRequestInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
torque2DipoleObj.tamSensorBodyInMsg.subscribeTo(tamCommObj.tamOutMsg)

dipoleMappingObj.dipoleRequestBodyInMsg.subscribeTo(
    torque2DipoleObj.dipoleRequestOutMsg)
dipoleMappingObj.mtbArrayConfigParamsInMsg.subscribeTo(mtbParamsInMsg)

rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)
rwMotorTorqueObj.vehControlInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)

rwStateEffector.rwMotorCmdInMsg.subscribeTo(
    rwMotorTorqueObj.rwMotorTorqueOutMsg)

mtbEff.mtbCmdInMsg.subscribeTo(dipoleMappingObj.dipoleRequestMtbOutMsg)
mtbEff.mtbParamsInMsg.subscribeTo(mtbParamsInMsg)
mtbEff.magInMsg.subscribeTo(magModule.envOutMsgs[0])

# ── 18. Initial conditions — ISS orbit ────────────────────────────────
oe = orbitalMotion.ClassicElements()
oe.a = (6371.0 + 408.0) * 1000.0
oe.e = 0.0005
oe.i = 51.6 * macros.D2R
oe.Omega = 30.0 * macros.D2R
oe.omega = 0.0 * macros.D2R
oe.f = 0.0 * macros.D2R
rN, vN = orbitalMotion.elem2rv(mu, oe)
scObject.hub.r_CN_NInit = rN
scObject.hub.v_CN_NInit = vN
scObject.hub.sigma_BNInit = [[0.3], [-0.2], [0.1]]
scObject.hub.omega_BN_BInit = [[0.5*macros.D2R], [-1.0*macros.D2R],
                                [2.0*macros.D2R]]

# ── 19. Data recorders ────────────────────────────────────────────────
samplingTime = unitTestSupport.samplingTime(SIM_TIME, DT, 400)
attErrLog = attError.attGuidOutMsg.recorder(samplingTime)
rwSpeedLog = rwStateEffector.rwSpeedOutMsg.recorder(samplingTime)
rwMotorLog = rwMotorTorqueObj.rwMotorTorqueOutMsg.recorder(samplingTime)
mtbCmdLog  = dipoleMappingObj.dipoleRequestMtbOutMsg.recorder(samplingTime)
for rec in [attErrLog, rwSpeedLog, rwMotorLog, mtbCmdLog]:
    scSim.AddModelToTask(simTaskName, rec)

# ── 20. Execute ───────────────────────────────────────────────────────
scSim.InitializeSimulation()
scSim.ConfigureStopTime(SIM_TIME)

t_start = time.perf_counter()
scSim.ExecuteSimulation()
elapsed = time.perf_counter() - t_start

import sys
print(f"Basilisk — elapsed: {elapsed:.2f} s  (sim time: 5400 s)", flush=True)

# Retrieve data before any cleanup
try:
    timeMin = attErrLog.times() * macros.NANO2MIN
    sigma_BR = np.array(attErrLog.sigma_BR)
    np.savez(os.path.join(os.path.dirname(__file__), "results_basilisk.npz"),
             time=timeMin, sigma_BR=sigma_BR, elapsed=elapsed)
    print("Results saved.", flush=True)
except Exception as e:
    print(f"Save failed: {e}", flush=True)
