extends RefCounted
## Double-precision vector/quaternion helpers.
##
## GDScript `float` is 64-bit but Vector3/Basis/Quaternion are 32-bit in stock
## Godot. All positions stay in these helpers (PackedFloat64Array) until the
## camera position has been subtracted; only then are they turned into Vector3.
##
## Frames: GCRS (x, y, z) maps to Godot (x, z, -y), a proper rotation, so
## Godot's Y-up convention is GCRS +z.

static func v3(x: float, y: float, z: float) -> PackedFloat64Array:
	return PackedFloat64Array([x, y, z])


static func sub(a: PackedFloat64Array, b: PackedFloat64Array) -> PackedFloat64Array:
	return PackedFloat64Array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])


static func add(a: PackedFloat64Array, b: PackedFloat64Array) -> PackedFloat64Array:
	return PackedFloat64Array([a[0] + b[0], a[1] + b[1], a[2] + b[2]])


static func scale(a: PackedFloat64Array, s: float) -> PackedFloat64Array:
	return PackedFloat64Array([a[0] * s, a[1] * s, a[2] * s])


static func dot(a: PackedFloat64Array, b: PackedFloat64Array) -> float:
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


static func cross(a: PackedFloat64Array, b: PackedFloat64Array) -> PackedFloat64Array:
	return PackedFloat64Array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


static func norm(a: PackedFloat64Array) -> float:
	return sqrt(dot(a, a))


static func unit(a: PackedFloat64Array) -> PackedFloat64Array:
	return scale(a, 1.0 / norm(a))


## Rotation matrix of a scalar-first Hamilton quaternion, row-major (9 values).
## Same as ADCS.helpers.math_helpers.rot_mat: v_A = R(q_A_B) v_B.
static func rot_mat(q: PackedFloat64Array) -> PackedFloat64Array:
	var w := q[0]
	var x := q[1]
	var y := q[2]
	var z := q[3]
	return PackedFloat64Array([
		w * w + x * x - y * y - z * z, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
		2.0 * (x * y + w * z), w * w - x * x + y * y - z * z, 2.0 * (y * z - w * x),
		2.0 * (x * z - w * y), 2.0 * (y * z + w * x), w * w - x * x - y * y + z * z,
	])


static func mat_vec(m: PackedFloat64Array, v: PackedFloat64Array) -> PackedFloat64Array:
	return PackedFloat64Array([
		m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
		m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
		m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
	])


static func mat_t_vec(m: PackedFloat64Array, v: PackedFloat64Array) -> PackedFloat64Array:
	return PackedFloat64Array([
		m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
		m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
		m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
	])


static func transpose(m: PackedFloat64Array) -> PackedFloat64Array:
	return PackedFloat64Array([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]])


## GCRS vector -> Godot vector (float32): (x, y, z) -> (x, z, -y).
static func to_godot(v: PackedFloat64Array) -> Vector3:
	return Vector3(v[0], v[2], -v[1])


## Godot Basis for an object whose local axes map into GCRS by `m` (row-major):
## columns are the GCRS images of the local axes, mapped into Godot.
static func basis_from(m: PackedFloat64Array) -> Basis:
	return Basis(
		to_godot(PackedFloat64Array([m[0], m[3], m[6]])),
		to_godot(PackedFloat64Array([m[1], m[4], m[7]])),
		to_godot(PackedFloat64Array([m[2], m[5], m[8]])))


## Camera rotation (local -> GCRS, row-major) looking along `forward` with `up_hint`.
## Godot cameras look down local -Z with +Y up.
static func look_rot(forward: PackedFloat64Array, up_hint: PackedFloat64Array) -> PackedFloat64Array:
	var back := scale(unit(forward), -1.0)
	var right := unit(cross(up_hint, back))
	var up := cross(back, right)
	return PackedFloat64Array([right[0], up[0], back[0], right[1], up[1], back[1], right[2], up[2], back[2]])


static func slerp(q0: PackedFloat64Array, q1: PackedFloat64Array, a: float) -> PackedFloat64Array:
	var d := q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]
	var b := q1
	if d < 0.0:
		b = PackedFloat64Array([-q1[0], -q1[1], -q1[2], -q1[3]])
		d = -d
	var out := PackedFloat64Array([0.0, 0.0, 0.0, 0.0])
	if d > 1.0 - 1e-12:
		for i in 4:
			out[i] = (1.0 - a) * q0[i] + a * b[i]
	else:
		var th := acos(minf(d, 1.0))
		var s0 := sin((1.0 - a) * th) / sin(th)
		var s1 := sin(a * th) / sin(th)
		for i in 4:
			out[i] = s0 * q0[i] + s1 * b[i]
	var n := sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3])
	for i in 4:
		out[i] /= n
	return out



## Hamilton product of scalar-first quaternions.
static func qmul(p: PackedFloat64Array, q: PackedFloat64Array) -> PackedFloat64Array:
	return PackedFloat64Array([
		p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
		p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
		p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
		p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
	])


## Attitude a fraction `a` of the way from q0 to q1, samples h seconds apart,
## following the sampled body rates w0, w1 [rad/s]. Slerp takes the short arc,
## so a body that turns more than 180 degrees per sample plays back slowly or
## backwards. This integrates dq/dt = q * (0, w) / 2 (as in ADCS; q is
## GCRS <- body, scalar first) with w linear in time between the samples, then
## spreads the remaining mismatch with q1 evenly over the interval, so the
## result starts on q0 and ends on q1. With zero rates it is slerp; for a
## constant-axis spin it is exact at any rate.
static func rate_interp(q0: PackedFloat64Array, q1: PackedFloat64Array, w0: PackedFloat64Array,
		w1: PackedFloat64Array, h: float, a: float) -> PackedFloat64Array:
	if h <= 0.0:
		return slerp(q0, q1, a)
	# About 10 degrees per substep; constant-axis turns are exact at any size,
	# so the cap only costs accuracy for very fast tumbles.
	var turn := norm(scale(add(w0, w1), 0.5 * h))
	var n := clampi(int(ceil(turn / deg_to_rad(10.0))), 4, 64)
	var end := _integrate_rates(q0, w0, w1, h, 1.0, n)
	var residual := rot_vec(qmul(PackedFloat64Array([end[0], -end[1], -end[2], -end[3]]), q1))
	var out := qmul(_integrate_rates(q0, w0, w1, h, a, n), rot_exp(scale(residual, a)))
	var m := sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3])
	return PackedFloat64Array([out[0] / m, out[1] / m, out[2] / m, out[3] / m])


## q0 carried through the fraction `a` of an h-second interval by body rates
## varying linearly from w0 to w1 (midpoint rate per substep, n substeps for
## the whole interval).
static func _integrate_rates(q0: PackedFloat64Array, w0: PackedFloat64Array, w1: PackedFloat64Array,
		h: float, a: float, n: int) -> PackedFloat64Array:
	var steps := maxi(1, int(ceil(n * a)))
	var dt := a * h / steps
	var q := q0
	for i in steps:
		var w := add(w0, scale(sub(w1, w0), (i + 0.5) * dt / h))
		q = qmul(q, rot_exp(scale(w, dt)))
	return q


## Quaternion of the rotation vector v [rad].
static func rot_exp(v: PackedFloat64Array) -> PackedFloat64Array:
	var angle := norm(v)
	if angle < 1e-15:
		return PackedFloat64Array([1.0, 0.0, 0.0, 0.0])
	var s := sin(0.5 * angle) / angle
	return PackedFloat64Array([cos(0.5 * angle), v[0] * s, v[1] * s, v[2] * s])


## Rotation vector [rad] of q, taking the short arc (angle in [0, pi]).
static func rot_vec(q: PackedFloat64Array) -> PackedFloat64Array:
	var w := q[0]
	var v := PackedFloat64Array([q[1], q[2], q[3]])
	if w < 0.0:
		w = -w
		v = scale(v, -1.0)
	var s := norm(v)
	if s < 1e-15:
		return PackedFloat64Array([0.0, 0.0, 0.0])
	return scale(v, 2.0 * atan2(s, w) / s)

## Cubic Hermite position between samples (r0, v0) and (r1, v1) h seconds apart, s in [0, 1].
static func hermite(r0: PackedFloat64Array, v0: PackedFloat64Array, r1: PackedFloat64Array,
		v1: PackedFloat64Array, h: float, s: float) -> PackedFloat64Array:
	var h00 := 2.0 * s * s * s - 3.0 * s * s + 1.0
	var h10 := s * s * s - 2.0 * s * s + s
	var h01 := -2.0 * s * s * s + 3.0 * s * s
	var h11 := s * s * s - s * s
	var out := PackedFloat64Array([0.0, 0.0, 0.0])
	for i in 3:
		out[i] = h00 * r0[i] + h10 * h * v0[i] + h01 * r1[i] + h11 * h * v1[i]
	return out
