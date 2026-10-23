extends SceneTree
## Headless unit tests for the viewer core.
##
##   godot --headless --path viewer --script res://tests/run_tests.gd -- [--fixture DIR]
##
## --fixture: an adcs-vizlog written by testing/test_viz/test_viewer_godot.py
## (a 7000 km circular orbit with known values). Prints "ALL PASSED" and exits 0
## on success.

const D := preload("res://src/core/dmath.gd")
const FrameStore := preload("res://src/core/frame_store.gd")
const Clock := preload("res://src/core/viz_clock.gd")
const Origin := preload("res://src/core/origin.gd")
const Protocol := preload("res://src/io/protocol.gd")
const FileSource := preload("res://src/io/file_source.gd")
const World := preload("res://src/scene/world.gd")
const Registry := preload("res://src/overlays/registry.gd")
const TF := preload("res://src/ui/timeline/time_fmt.gd")
const Ticks := preload("res://src/ui/timeline/ticks.gd")
const Lanes := preload("res://src/ui/timeline/lanes.gd")
const Bookmarks := preload("res://src/ui/timeline/bookmarks.gd")
const Att := preload("res://src/ui/timeline/attitude.gd")
const Timeline := preload("res://src/ui/timeline/timeline.gd")


class FakeWorld:
	extends RefCounted
	var info := {"k_earth": 0.5}


class FakeSource:
	extends RefCounted
	var store: RefCounted
	var live := false
	var manifest := {"epoch": {"iso_utc": "2026-03-14T23:59:00.000Z"}}
	var scene := {}
	var run_id := "0"

var failures := PackedStringArray()
var checks := 0


func _init() -> void:
	var args := {}
	var argv := OS.get_cmdline_user_args()
	for i in argv.size():
		if argv[i] == "--fixture" and i + 1 < argv.size():
			args["fixture"] = argv[i + 1]
	for t: Callable in [test_rot_mat, test_look_rot, test_slerp, test_hermite, test_frame_store, test_clock,
			test_origin, test_protocol_frames, test_moon_frame, test_arrow_lengths, test_tracks,
			test_time_format, test_ticks, test_eclipse_lane, test_pass_lane, test_bookmarks, test_attitude,
			test_timeline_mapping]:
		t.call()
	if args.has("fixture"):
		test_file_source(args["fixture"])
	if failures.is_empty():
		print("ALL PASSED (%d checks)" % checks)
		quit(0)
	else:
		for f in failures:
			printerr("FAIL: ", f)
		quit(1)


func expect(cond: bool, what: String) -> void:
	checks += 1
	if not cond:
		failures.append(what)


func near(a: float, b: float, tol: float, what: String) -> void:
	expect(absf(a - b) <= tol, "%s: %s vs %s (tol %s)" % [what, a, b, tol])


# ------------------------------------------------------------------ dmath
func test_rot_mat() -> void:
	var m := D.rot_mat(PackedFloat64Array([1, 0, 0, 0]))
	for i in 9:
		near(m[i], 1.0 if i in [0, 4, 8] else 0.0, 1e-15, "identity quaternion [%d]" % i)
	# 90 deg about +z maps x -> y (active, scalar-first), as ADCS.helpers.math_helpers.rot_mat.
	var q := PackedFloat64Array([cos(PI / 4), 0, 0, sin(PI / 4)])
	var v := D.mat_vec(D.rot_mat(q), D.v3(1, 0, 0))
	near(v[0], 0.0, 1e-15, "rotz90 x")
	near(v[1], 1.0, 1e-15, "rotz90 y")
	# GCRS -> Godot mapping is a proper rotation: z up becomes Godot +Y.
	expect(D.to_godot(D.v3(0, 0, 1)).is_equal_approx(Vector3.UP), "GCRS +z is Godot up")
	expect(is_equal_approx(D.basis_from(D.rot_mat(q)).determinant(), 1.0), "basis_from keeps handedness")


func test_look_rot() -> void:
	var r := D.look_rot(D.v3(1, 2, 3), D.v3(0, 0, 1))
	var back := PackedFloat64Array([r[2], r[5], r[8]])
	var f := D.unit(D.v3(1, 2, 3))
	near(D.dot(back, f), -1.0, 1e-12, "camera looks along -Z")
	near(D.dot(PackedFloat64Array([r[0], r[3], r[6]]), PackedFloat64Array([r[1], r[4], r[7]])), 0.0, 1e-12,
		"orthogonal axes")


func test_slerp() -> void:
	var q0 := PackedFloat64Array([1, 0, 0, 0])
	var q1 := PackedFloat64Array([cos(PI / 4), 0, 0, sin(PI / 4)])
	var h := D.slerp(q0, q1, 0.5)
	near(h[0], cos(PI / 8), 1e-12, "slerp halfway w")
	var neg := PackedFloat64Array([-q1[0], -q1[1], -q1[2], -q1[3]])
	var h2 := D.slerp(q0, neg, 0.5)
	near(absf(h2[0]), cos(PI / 8), 1e-12, "slerp takes the short arc with -q")


func test_hermite() -> void:
	var a := 7000.0e3
	var n := sqrt(3.986004418e14 / pow(a, 3))
	var h := 30.0
	var r0 := D.v3(a, 0, 0)
	var v0 := D.v3(0, a * n, 0)
	var r1 := D.v3(a * cos(n * h), a * sin(n * h), 0)
	var v1 := D.v3(-a * n * sin(n * h), a * n * cos(n * h), 0)
	var mid := D.hermite(r0, v0, r1, v1, h, 0.5)
	var exact := D.v3(a * cos(n * h / 2), a * sin(n * h / 2), 0)
	expect(D.norm(D.sub(mid, exact)) < 0.05, "Hermite within 5 cm at 30 s spacing (%f m)" % D.norm(D.sub(mid, exact)))


# ------------------------------------------------------------------ frame store, clock, origin
func test_frame_store() -> void:
	var fs := FrameStore.new(PackedStringArray(["t_s", "x"]), PackedInt32Array([1, 2]))
	fs.append_block({"t_s": PackedFloat64Array([0.0, 1.0]), "x": PackedFloat64Array([0, 0, 10, 20])}, 2)
	fs.append_row(PackedFloat64Array([2.0, 20, 40]))
	expect(fs.n_rows == 3, "rows appended (packed arrays written back)")
	var v := fs.value("x", 1.5)
	near(v[0], 15.0, 1e-12, "linear interpolation")
	near(fs.value("x", 99.0)[1], 40.0, 1e-12, "clamped past the end")
	var ks: Array = fs.locate(1.0)
	expect(ks[0] == 0 and is_equal_approx(ks[1], 1.0) or ks[0] == 1 and is_equal_approx(ks[1], 0.0), "locate at a sample")


func test_clock() -> void:
	var c := Clock.new()
	c.set_range(0.0, 10.0)
	c.rate = 2.0
	c.advance(1.0)
	near(c.t, 2.0, 1e-12, "advance x rate")
	c.advance(100.0)
	expect(c.t == 10.0 and c.paused, "stops at the end")
	c.toggle_pause()
	expect(c.t == 0.0 and not c.paused, "play again from the start")
	c.seek(-5.0)
	expect(c.t == 0.0, "seek clamps")
	c.faster()
	expect(c.rate > 2.0, "faster")


func test_origin() -> void:
	var o := Origin.new()
	expect(o.begin_frame(D.v3(0, 0, 0), 1.0, 0.5), "craft 0.5 m away is near")
	near(o.near, 0.1, 1e-12, "near = 0.2 x surface distance")
	near(o.far / o.near, Origin.FRUSTUM_RATIO, 1e-6, "far/near ratio")
	expect(not o.begin_frame(D.v3(0, 0, 0), 1.0e6, 1.0), "craft 1000 km away is in scaled space")
	var last := 0.0
	for d: float in [1.0e3, 1.0e4, 2.0e4, 1.0e6, 7.0e6, 4.0e8, 1.0e10]:
		var drawn := o.far_scale(d) * d
		expect(drawn > last and drawn < o.far, "far_scale is monotonic and inside the frustum at %s m" % d)
		last = drawn


func test_moon_frame() -> void:
	var r := World.moon_rotation(D.v3(3.8e8, 1.0e7, -2.0e7))
	var x := PackedFloat64Array([r[0], r[3], r[6]])
	near(D.dot(x, D.unit(D.v3(-3.8e8, -1.0e7, 2.0e7))), 1.0, 1e-12, "Moon +X faces Earth")


# ------------------------------------------------------------------ protocol
func _frame(kind: int, payload: PackedByteArray, seq: int) -> PackedByteArray:
	var h := StreamPeerBuffer.new()
	h.big_endian = false
	h.put_data("ADVZ".to_ascii_buffer())
	h.put_u16(1)
	h.put_u16(kind)
	h.put_u32(payload.size())
	h.put_u64(seq)
	var out := h.data_array
	out.append_array(payload)
	return out


func test_protocol_frames() -> void:
	var c := Protocol.new()
	var hello := {"protocol": 1, "mode": "playback", "scene": {},
		"manifest": {"channels": [{"name": "t_s", "width": 1}, {"name": "r", "width": 3}], "runs": [{"id": "0"}]}}
	var buf := _frame(Protocol.Msg.HELLO, JSON.stringify(hello).to_utf8_buffer(), 0)
	var row := StreamPeerBuffer.new()
	row.big_endian = false
	row.put_u16(1)
	row.put_data("0".to_utf8_buffer())
	for x: float in [5.0, 1.0, 2.0, 3.0]:
		row.put_double(x)
	buf.append_array(_frame(Protocol.Msg.ROW, row.data_array, 1))
	var asset := StreamPeerBuffer.new()
	asset.big_endian = false
	asset.put_u16(5)
	asset.put_data("a.glb".to_utf8_buffer())
	var meta := JSON.stringify({"kind": "model_lod"}).to_utf8_buffer()
	asset.put_u32(meta.size())
	asset.put_data(meta)
	asset.put_data(PackedByteArray([1, 2, 3]))
	buf.append_array(_frame(Protocol.Msg.ASSET, asset.data_array, 2))
	buf.append_array(_frame(Protocol.Msg.END, JSON.stringify({}).to_utf8_buffer(), 3))
	# Deliver everything in one read, half of the last frame first (partial frame handling).
	c._buf = buf.slice(0, buf.size() - 3)
	var msgs := c.poll_messages()
	expect(msgs.size() == 3, "partial final frame waits (got %d)" % msgs.size())
	expect(c.names == PackedStringArray(["t_s", "r"]), "HELLO sets the channel layout")
	var decoded: Array = c.decode_row(msgs[1][2])
	expect(decoded[0] == "0" and (decoded[1] as PackedFloat64Array) == PackedFloat64Array([5, 1, 2, 3]), "ROW decodes")
	expect(c.assets.has("a.glb") and (c.assets["a.glb"]["data"] as PackedByteArray) == PackedByteArray([1, 2, 3]),
		"ASSET stored")
	c._buf.append_array(buf.slice(buf.size() - 3))
	c.poll_messages()
	expect(c.session_ended, "END {} ends the session")


# ------------------------------------------------------------------ file source
# ------------------------------------------------------------------ overlays
func _arrow_ctx(store: RefCounted, rot: PackedFloat64Array) -> Dictionary:
	var f := {"r_sc": D.v3(7.0e6, 0, 0), "r_sun": D.v3(7.0e6, 1.5e11, 0), "v_sc": D.v3(0, 7500, 0)}
	return {"frame": f, "store": store, "d_craft": 2.0, "k_craft": 1.0, "focal": 1000.0, "rot": rot,
		"t": 0.0, "frames": {"tau": "BODY", "b": "GCRS"}}


func test_arrow_lengths() -> void:
	var store := FrameStore.new(PackedStringArray(["t_s", "tau", "b"]), PackedInt32Array([1, 3, 3]))
	store.append_row(PackedFloat64Array([0.0, 100.0, 0, 0, 1, 0, 0]))
	var rotz := D.rot_mat(PackedFloat64Array([cos(PI / 4), 0, 0, sin(PI / 4)]))  # body -> GCRS: 90 deg about z
	var ctx := _arrow_ctx(store, rotz)
	var cases := [  # descriptor, expected length [px], expected BODY direction
		[{"source": "tau", "scale": "log", "ref": 1.0}, 24.0 + 40.0 * 2.0, Vector3(1, 0, 0)],
		[{"source": "tau", "scale": "log", "ref": 1.0e-6}, 180.0, Vector3(1, 0, 0)],  # clamped
		[{"source": "tau", "scale": "physical", "units_per_m": 50.0}, 2.0 * 1000.0 / 2.0, Vector3(1, 0, 0)],
		[{"source": "b", "scale": "fixed_px", "px": 90.0}, 90.0, Vector3(0, -1, 0)],  # GCRS +x = body -y
		[{"source": "@sun", "scale": "fixed_px", "px": 60.0}, 60.0, Vector3(1, 0, 0)],  # GCRS +y -> body +x
	]
	for c: Array in cases:
		var d: Dictionary = c[0]
		d["type"] = "arrow"
		var a: Node3D = Registry.create(d, {})
		a.update(ctx)
		var info: Dictionary = a.diagnostics()
		near(info["length_px"], c[1], 1e-6, "arrow %s length" % d)
		var dir: Array = info["dir_body"]
		near(Vector3(dir[0], dir[1], dir[2]).distance_to(c[2]), 0.0, 1e-6, "arrow %s direction" % d)
		# Drawn length on screen: local metres x focal / distance.
		near(a.mesh_node.basis.y.length() * 1000.0 / 2.0, c[1], 1e-3, "arrow %s mesh length" % d)
		a.free()


func test_tracks() -> void:
	var store := FrameStore.new(PackedStringArray(["t_s", "r_sc", "q_itrs_gcrs"]), PackedInt32Array([1, 3, 4]))
	var qz := PackedFloat64Array([cos(PI / 4), 0, 0, sin(PI / 4)])
	for row: Array in [[0.0, 7000.0, 0.0], [100.0, 0.0, 7000.0], [200.0, -7000.0, 0.0]]:
		store.append_row(PackedFloat64Array([row[0], row[1], row[2], 0.0]) + qz)
	var cam := D.v3(7.0e6 + 1.0, 0, 0)
	var ctx := {"store": store, "t": 0.0, "cam_pos": cam, "world": FakeWorld.new(),
		"frame": {"r_sc": D.v3(7.0e6, 0, 0), "q_itrs_gcrs": qz}}
	for kind: String in ["orbit", "ground"]:
		var tr: Node3D = Registry.create({"type": "polyline", "track": kind, "past_s": 1000.0, "future_s": 1000.0}, {})
		tr.update(ctx)
		var verts: PackedVector3Array = tr.mesh.surface_get_arrays(0)[Mesh.ARRAY_VERTEX]
		expect(verts.size() == 3, "%s track: one vertex per row" % kind)
		var xf: Transform3D = tr.mesh_node.transform
		for i in verts.size():
			var r := D.scale(store.get_row("r_sc", i), 1000.0)
			if kind == "ground":  # Earth-fixed point above the sub-satellite point (constant rotation here)
				r = D.scale(D.unit(r), 6378137.0 + 3000.0)
			var want := D.to_godot(D.scale(D.sub(r, cam), 0.5))
			var tol := 1e-4 if i == 0 and kind == "orbit" else 2.0  # metre-level far away, exact near the anchor
			near((xf * verts[i]).distance_to(want), 0.0, tol, "%s track vertex %d" % [kind, i])
		tr.free()


# ------------------------------------------------------------------ timeline
func test_time_format() -> void:
	var e := TF.parse_iso("2026-03-14T23:59:00.000Z")
	expect(TF.utc(e, 0) == "2026-03-14 23:59:00", "utc format")
	expect(TF.utc(e + 61.25, 1) == "2026-03-15 00:00:01.3" or TF.utc(e + 61.25, 1) == "2026-03-15 00:00:01.2", "utc rounding across midnight")
	expect(TF.utc(e + 59.9996, 3) == "2026-03-15 00:00:00.000", "ms rounding carries into the seconds")
	expect(TF.elapsed(3723.0) == "T+01:02:03", "elapsed %s" % TF.elapsed(3723.0))
	expect(TF.elapsed(2 * 86400.0 + 5.0) == "T+2d 00:00:05", "elapsed days")
	expect(TF.elapsed(-5.0) == "T−00:00:05", "negative elapsed")
	near(TF.parse("T+1:02:03", e, 0.0), 3723.0, 1e-9, "parse T+h:m:s")
	near(TF.parse("+2460", e, 0.0), 2460.0, 1e-9, "parse +seconds")
	near(TF.parse("T+1h2m", e, 0.0), 3720.0, 1e-9, "parse T+1h2m")
	near(TF.parse("90s", e, 0.0), 90.0, 1e-9, "parse plain duration")
	near(TF.parse("2026-03-15 00:01:00", e, 0.0), 120.0, 1e-6, "parse full UTC")
	near(TF.parse("2026-03-15T00:01:00.5Z", e, 0.0), 120.5, 1e-6, "parse ISO with fraction")
	near(TF.parse("23:59:30", e, 0.0), 30.0, 1e-6, "parse clock time on the date shown")
	near(TF.parse("00:00:30", e, 120.0), 90.0, 1e-6, "clock time after midnight uses the shown date")
	expect(is_nan(TF.parse("soon", e, 0.0)), "garbage is NAN")
	expect(is_nan(TF.parse("25:00", e, 0.0)), "bad clock time is NAN")


func test_ticks() -> void:
	var e := TF.parse_iso("2026-03-14T23:59:00Z")
	var ticks := Ticks.make(0.0, 600.0, e, 1000.0, 7.0)  # 10 min over 1000 px
	var majors := ticks.filter(func(k: Dictionary) -> bool: return k["major"])
	expect(majors.size() >= 4 and majors.size() <= 12, "a readable number of labels (%d)" % majors.size())
	for k: Dictionary in majors:
		var u: float = e + float(k["t"])
		expect(absf(fmod(u, 60.0)) < 1e-6, "major ticks on whole minutes")
	expect(majors.any(func(k: Dictionary) -> bool: return k["label"] == "Mar 15"), "midnight labelled with the date")
	var step: float = Ticks.STEPS[Ticks.choose(600.0, 1000.0, 7.0)][0]
	expect(step * 1000.0 / 600.0 >= Ticks.label_px(step, 7.0), "labels fit")
	var deep := Ticks.make(0.0, 2.0, e, 1000.0, 7.0)
	expect(deep.size() > 5 and String(deep.filter(func(k: Dictionary) -> bool: return k["major"])[0]["label"]).length() == 8,
		"seconds shown when zoomed in")


func _store(cols: Dictionary, n: int) -> RefCounted:
	var names := PackedStringArray(cols.keys())
	var widths := PackedInt32Array()
	for nm in names:
		widths.append((cols[nm] as PackedFloat64Array).size() / n)
	var fs := FrameStore.new(names, widths)
	fs.append_block(cols, n)
	return fs


func test_eclipse_lane() -> void:
	var t := PackedFloat64Array()
	var sf := PackedFloat64Array()
	for k in 11:
		t.append(10.0 * k)
	sf.append_array([1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
	var l := Lanes.new()
	l.setup(_store({"t_s": t, "shadow_factor": sf}, 11), {})
	l.update()
	var spans := l.eclipse_spans()
	expect(spans.size() == 5, "sunlit, penumbra, umbra, penumbra, sunlit (%d)" % spans.size())
	expect(int(spans[2][2]) == Lanes.Light.UMBRA, "umbra in the middle")
	near(spans[0][1], 20.0 + 10.0 * (0.999 - 1.0) / (0.5 - 1.0), 1e-9, "penumbra entry at the 0.999 crossing")
	near(spans[2][0], 30.0 + 10.0 * (0.001 - 0.5) / (0.0 - 0.5), 1e-9, "umbra entry at the 0.001 crossing")
	near(spans[4][1], 100.0, 1e-12, "last span ends at the last row")
	expect(int(l.at(50.0)["sky"]) == Lanes.Light.UMBRA, "state at a time")


func test_pass_lane() -> void:
	# Equatorial circular orbit, Earth frame = GCRS (identity rotation), site on the equator at lon 0:
	# elevation crosses 10 deg symmetrically about t = 0 (spacecraft overhead).
	var n := 121
	var t := PackedFloat64Array()
	var r := PackedFloat64Array()
	var q := PackedFloat64Array()
	var rad := 6878.137
	var w := 0.0011  # rad/s
	for k in n:
		var ts := -600.0 + 10.0 * k
		t.append(ts)
		r.append_array([rad * cos(w * ts), rad * sin(w * ts), 0.0])
		q.append_array([1.0, 0.0, 0.0, 0.0])
	var site := {"name": "eq", "itrs_m": [6378137.0, 0.0, 0.0], "up_itrs": [1.0, 0.0, 0.0], "min_elev_deg": 10.0}
	var l := Lanes.new()
	l.setup(_store({"t_s": t, "r_sc": r, "q_itrs_gcrs": q}, n), {"sites": [site]})
	l.update()
	var ps := l.passes(0)
	expect(ps.size() == 1 and not ps[0][4], "one complete pass")
	# Analytic rise time: elevation(theta) = 10 deg where theta = w t.
	var theta := 0.0
	var lo := 0.0
	var hi := PI / 2.0
	for i in 60:
		theta = 0.5 * (lo + hi)
		var los := Vector2(rad * 1000.0 * cos(theta) - 6378137.0, rad * 1000.0 * sin(theta))
		if rad_to_deg(asin(los.x / los.length())) > 10.0:
			lo = theta
		else:
			hi = theta
	var t_set := theta / w
	near(ps[0][1], t_set, 1.5, "set time within the row interpolation error")
	near(ps[0][0], -t_set, 1.5, "rise time symmetric")
	near(ps[0][2], 0.0, 1e-9, "peak overhead")
	near(ps[0][3], 90.0, 0.01, "peak elevation 90 deg")


func test_bookmarks() -> void:
	var dir := "user://test_bookmarks_%d" % Time.get_ticks_usec()
	var scene := {"timeline": {"bookmarks": [{"id": "b0", "t_s": 5.0, "label": "from log"}]}}
	var man := {"epoch": {"jd_tt_whole": 2460000.0}}
	var b := Bookmarks.new()
	b.setup(man, scene, "0", PackedFloat64Array([1, 2, 3]), dir)
	expect(b.items.size() == 1 and b.items[0]["locked"], "log bookmark loaded, locked")
	var mine := b.add(2.0, "mine")
	b.rename("b0", "renamed")  # locked: ignored
	b.remove("b0")
	expect(b.items.size() == 2 and b.items[0]["id"] == mine["id"], "sorted, locked kept")
	var again := Bookmarks.new()
	again.setup(man, scene, "0", PackedFloat64Array([1, 2, 3]), dir)
	expect(again.items.size() == 2 and again.find(mine["id"]).get("label") == "mine", "own bookmarks saved and reloaded")
	var other := Bookmarks.new()
	other.setup(man, scene, "1", PackedFloat64Array([1, 2, 3]), dir)
	expect(other.items.size() == 1, "another run has its own bookmarks")
	expect(again.add(3.0)["id"] != mine["id"], "ids stay unique after reload")
	DirAccess.remove_absolute(again.path)
	DirAccess.remove_absolute(dir)


func test_attitude() -> void:
	var r := D.v3(7.0e6, 0, 0)
	var v := D.v3(0, 7500, 0)
	# LVLH here: Z = -x, Y = -(r x v)/|..| = -z, X = Y x Z = +y. Body = LVLH -> R_gcrs_body columns = LVLH axes.
	var m := PackedFloat64Array([0, 0, -1, 1, 0, 0, 0, -1, 0])  # columns X=(0,1,0), Y=(0,0,-1), Z=(-1,0,0)
	var q := _quat_from_mat(m)
	var rpy := Att.rpy_lvlh_deg(r, v, q)
	for i in 3:
		near(rpy[i], 0.0, 1e-9, "body aligned with LVLH: angle %d" % i)
	# Yaw 30 deg about LVLH Z.
	var c := cos(deg_to_rad(30.0))
	var s := sin(deg_to_rad(30.0))
	var yawed := PackedFloat64Array()  # R_gcrs_body = R_gcrs_lvlh * Rz(30)
	var rz := PackedFloat64Array([c, -s, 0, s, c, 0, 0, 0, 1])
	yawed.resize(9)
	for i in 3:
		for j in 3:
			yawed[3 * i + j] = m[3 * i] * rz[j] + m[3 * i + 1] * rz[3 + j] + m[3 * i + 2] * rz[6 + j]
	rpy = Att.rpy_lvlh_deg(r, v, _quat_from_mat(yawed))
	near(rpy[2], 30.0, 1e-9, "yaw 30")
	near(rpy[0], 0.0, 1e-9, "roll 0")
	near(rpy[1], 0.0, 1e-9, "pitch 0")


static func _quat_from_mat(m: PackedFloat64Array) -> PackedFloat64Array:
	var tr := m[0] + m[4] + m[8]
	var w := sqrt(maxf(0.0, 1.0 + tr)) / 2.0
	if w > 1e-6:
		return PackedFloat64Array([w, (m[7] - m[5]) / (4 * w), (m[2] - m[6]) / (4 * w), (m[3] - m[1]) / (4 * w)])
	var x := sqrt(maxf(0.0, 1.0 + m[0] - m[4] - m[8])) / 2.0
	if x > 1e-6:
		return PackedFloat64Array([(m[7] - m[5]) / (4 * x), x, (m[1] + m[3]) / (4 * x), (m[2] + m[6]) / (4 * x)])
	var y := sqrt(maxf(0.0, 1.0 - m[0] + m[4] - m[8])) / 2.0
	return PackedFloat64Array([(m[2] - m[6]) / (4 * y), (m[1] + m[3]) / (4 * y), y, (m[5] + m[7]) / (4 * y)])


func test_timeline_mapping() -> void:
	var t := PackedFloat64Array()
	var r := PackedFloat64Array()
	for k in 101:
		t.append(10.0 * k)
		r.append_array([7000.0, 0.0, 0.0])
	var src := FakeSource.new()
	src.store = _store({"t_s": t, "r_sc": r}, 101)
	var clock := Clock.new()
	clock.set_range(0.0, 1000.0)
	var tl: Control = Timeline.new()
	tl.setup(clock, src)
	tl._rects = {"track": Rect2(100, 0, 1000, 40)}
	tl.set_view(0.0, 1000.0)
	expect(tl.fit, "whole run = fit")
	near(tl.t_to_x(500.0), 600.0, 1e-9, "t -> x")
	near(tl.x_to_t(350.0), 250.0, 1e-9, "x -> t")
	tl.zoom_at(250.0, 0.5)
	near(tl.view1 - tl.view0, 500.0, 1e-9, "zoom halves the span")
	near(tl.x_to_t(tl.t_to_x(250.0)), 250.0, 1e-9, "anchor stays under the cursor")
	near(tl.t_to_x(250.0), 350.0, 1e-9, "anchor at the same pixel")
	tl.zoom_at(0.0, 1e-9)
	near(tl.view1 - tl.view0, Timeline.MIN_SPAN, 1e-9, "zoom stops at the minimum span")
	tl.set_view(0.0, 100.0)  # 10 px per second, rows every 10 s = 100 px apart
	near(tl.snap(41.0), 41.0, 1e-12, "no snap far from a sample")
	near(tl.snap(40.4), 40.0, 1e-12, "snap within a few pixels")
	tl.set_view(-50.0, 30.0)
	near(tl.view0, 0.0, 1e-12, "window kept inside the run")
	clock.t = 25.0
	tl.step_row(1)
	near(clock.t, 30.0, 1e-12, "next sample from between samples")
	tl.step_row(1)
	near(clock.t, 40.0, 1e-12, "next sample from a sample")
	tl.step_row(-1)
	near(clock.t, 30.0, 1e-12, "previous sample")
	tl.free()


func test_file_source(path: String) -> void:
	var fs := FileSource.new()
	expect(fs.open(path), "fixture vizlog opens: %s" % fs.error_message)
	if fs.store == null:
		return
	expect(fs.store.n_rows == 121, "fixture rows (%d)" % fs.store.n_rows)
	# The fixture is a 7000 km circular orbit: |r| at any (interpolated) time is 7000 km.
	for ts: float in [0.0, 37.3, 119.9]:
		var f := fs.store.sample(ts)
		near(D.norm(f["r_sc"]), 7.0e6, 0.05, "Hermite radius at t=%s" % ts)
	var q: PackedFloat64Array = fs.store.sample(60.0)["q_itrs_gcrs"]
	near(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], 1.0, 1e-12, "unit Earth orientation")
