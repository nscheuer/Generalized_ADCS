extends RefCounted
## The spacecraft's state at a moment, in the terms the timeline shows:
## attitude relative to LVLH as roll/pitch/yaw, body rates, pointing error.
##
## LVLH: Z toward nadir, Y along the negative orbit normal, X = Y x Z (along
## the velocity for a circular orbit). Roll/pitch/yaw: the 3-2-1 sequence
## (yaw about Z, then pitch about Y, then roll about X) taking LVLH to BODY,
## so all zero means the body axes are the LVLH axes.

const D := preload("res://src/core/dmath.gd")
const R_EARTH := 6378137.0


## Rows of R_lvlh_gcrs (v_lvlh = R v_gcrs) as a row-major 3x3, from r, v [m, m/s].
static func lvlh_rows(r: PackedFloat64Array, v: PackedFloat64Array) -> PackedFloat64Array:
	var z := D.unit(D.scale(r, -1.0))
	var y := D.unit(D.scale(D.cross(r, v), -1.0))
	var x := D.cross(y, z)
	return PackedFloat64Array([x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]])


## [roll, pitch, yaw] in degrees of the BODY frame relative to LVLH.
static func rpy_lvlh_deg(r: PackedFloat64Array, v: PackedFloat64Array, q_gcrs_body: PackedFloat64Array) -> PackedFloat64Array:
	var l := lvlh_rows(r, v)
	var b := D.rot_mat(q_gcrs_body)  # body -> GCRS
	var m := PackedFloat64Array()  # R_lvlh_body = R_lvlh_gcrs * R_gcrs_body
	m.resize(9)
	for i in 3:
		for j in 3:
			m[3 * i + j] = l[3 * i] * b[j] + l[3 * i + 1] * b[3 + j] + l[3 * i + 2] * b[6 + j]
	var pitch := -asin(clampf(m[6], -1.0, 1.0))
	var roll := atan2(m[7], m[8])
	var yaw := atan2(m[3], m[0])
	return PackedFloat64Array([rad_to_deg(roll), rad_to_deg(pitch), rad_to_deg(yaw)])


## Everything the state card and the export show, from a FrameStore sample (metres).
static func describe(f: Dictionary) -> Dictionary:
	var q: PackedFloat64Array = f["q_gcrs_body"]
	var w: PackedFloat64Array = f["w_body"]
	var out := {
		"t_s": f["t"],
		"q_gcrs_body": [q[0], q[1], q[2], q[3]],
		"rpy_lvlh_deg": Array(rpy_lvlh_deg(f["r_sc"], f["v_sc"], q)),
		"w_body_deg_s": [rad_to_deg(w[0]), rad_to_deg(w[1]), rad_to_deg(w[2])],
		"altitude_km": (D.norm(f["r_sc"]) - R_EARTH) / 1000.0,
		"shadow_factor": f["shadow_factor"],
	}
	if f.has("pointing_err"):
		var e: float = (f["pointing_err"] as PackedFloat64Array)[0]
		out["pointing_err_deg"] = rad_to_deg(e) if not is_nan(e) else null
	return out
