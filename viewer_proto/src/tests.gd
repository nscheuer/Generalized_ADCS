extends RefCounted
## Phase 0-V tests. Each returns a Dictionary of raw measurements; pass/fail is
## decided by tools/viz_validation/run_phase0.py against values computed
## independently in Python.

const D := preload("res://src/dmath.gd")
const R_EARTH := 6378137.0

var main  # main.gd instance (untyped: avoids a cyclic preload)
var _lin := PackedFloat32Array()


func _init(m: Variant) -> void:
	main = m
	for i in 256:
		_lin.append(Color(i / 255.0, 0, 0).srgb_to_linear().r)


func _cfg(name: String) -> Dictionary:
	return main.config.get(name, {})


func _image() -> Image:
	return main.scene.viewport.get_texture().get_image()


func _wait_frames(n: int) -> void:
	for i in n:
		await RenderingServer.frame_post_draw


func _hide_all() -> void:
	for n: Node3D in [main.scene.earth, main.scene.moon, main.scene.craft, main.scene.stars]:
		n.visible = false


# ---------------------------------------------------------------- (a1) jitter
## Camera locked 1 m off the body +X face looking back at it: the fiducial must
## not move on screen while the spacecraft flies at 7.5 km/s.
func jitter() -> Dictionary:
	var cfg := _cfg("jitter")
	var frames: int = cfg.get("frames", 600)
	var warmup: int = cfg.get("warmup", 10)
	var fps: float = cfg.get("fps", 60.0)
	main.scene.absolute = cfg.get("absolute", false)
	main.scene.earth.visible = true
	var size: Vector2i = main.scene.viewport.size
	var half := 140
	var cx := PackedFloat64Array()
	var cy := PackedFloat64Array()
	var lost := 0
	for i in frames + warmup:
		var t := maxf(0.0, (i - warmup) / fps)
		var f: Dictionary = main.store.sample(t)
		var rb := D.rot_mat(f["q_gcrs_body"])
		var xb := PackedFloat64Array([rb[0], rb[3], rb[6]])
		var zb := PackedFloat64Array([rb[2], rb[5], rb[8]])
		var cam := D.add(f["r_sc"], xb)
		main.scene.update(f, cam, D.look_rot(D.scale(xb, -1.0), zb))
		await RenderingServer.frame_post_draw
		if i < warmup:
			continue
		var c := _green_centroid(_image(), Rect2i(size.x / 2 - half, size.y / 2 - half, 2 * half, 2 * half))
		if c.x < 0.0:
			lost += 1
		cx.append(c.x)
		cy.append(c.y)
		if i == warmup:
			main.save_image(_image(), "jitter_first.png")
	return {"frames": frames, "absolute": main.scene.absolute, "cx": cx, "cy": cy, "lost": lost,
		"res": [size.x, size.y]}


## Intensity-weighted centroid of green-dominant pixels (linear light), or (-1, -1).
func _green_centroid(img: Image, roi: Rect2i) -> Vector2:
	img.convert(Image.FORMAT_RGBA8)
	var data := img.get_data()
	var w := img.get_width()
	var sw := 0.0
	var sx := 0.0
	var sy := 0.0
	for y in range(roi.position.y, roi.end.y):
		var row := y * w
		for x in range(roi.position.x, roi.end.x):
			var i := (row + x) * 4
			var g := _lin[data[i + 1]]
			var other := maxf(_lin[data[i]], _lin[data[i + 2]])
			var wgt := g - other
			if wgt > 0.02:
				sw += wgt
				sx += wgt * (x + 0.5)
				sy += wgt * (y + 0.5)
	if sw <= 0.0:
		return Vector2(-1.0, -1.0)
	return Vector2(sx / sw, sy / sw)


# ---------------------------------------------------------------- (a2) zoom & depth
func _probe_quads() -> Array[MeshInstance3D]:
	var quads: Array[MeshInstance3D] = []
	for color: Vector3 in [Vector3(1, 0, 0), Vector3(0, 0, 1)]:
		var mi := MeshInstance3D.new()
		mi.mesh = QuadMesh.new()
		var mat := ShaderMaterial.new()
		mat.shader = load("res://shaders/flat.gdshader")
		mat.set_shader_parameter("color", color)
		mi.material_override = mat
		main.scene.viewport.add_child(mi)
		quads.append(mi)
	return quads


func _center_color() -> String:
	var img := _image()
	var c := img.get_pixel(img.get_width() / 2, img.get_height() / 2)
	if c.r > 0.5 and c.g < 0.3 and c.b < 0.3:
		return "red"
	if c.b > 0.5 and c.r < 0.3 and c.g < 0.3:
		return "blue"
	if c.r > 0.5 and c.g > 0.5 and c.b > 0.5:
		return "white"
	return "black" if c.r + c.g + c.b < 0.05 else "other"


func zoom() -> Dictionary:
	var cfg := _cfg("zoom")
	_hide_all()
	var cam: Camera3D = main.scene.camera
	cam.basis = Basis()
	cam.position = Vector3.ZERO
	var quads := _probe_quads()

	# 1. Single-frustum limit: does a quad at 0.5 x far render at all?
	var ratio_probe := []
	quads[1].visible = false
	for ratio: float in [1e5, 1e6, 1e7, 1e8, 1e9]:
		cam.near = 1.0
		cam.far = ratio
		(quads[0].mesh as QuadMesh).size = Vector2.ONE * 0.1 * ratio
		quads[0].position = Vector3(0, 0, -0.5 * ratio)
		await _wait_frames(3)
		ratio_probe.append({"far_over_near": ratio, "visible": _center_color() == "red"})
	quads[1].visible = true

	# 2. Depth precision within one frustum (far/near = scene.FRUSTUM_RATIO), with
	#    the probe at 0.9 x far, the worst case for reverse-Z.
	var precision := []
	var rel_sep: float = cfg.get("rel_sep", 1e-5)
	for dist: float in cfg.get("distances", [1.0, 1e3, 1e6, 4e8]):
		cam.far = dist / 0.9
		cam.near = cam.far / main.scene.FRUSTUM_RATIO
		for front in 2:
			var back := 1 - front
			(quads[front].mesh as QuadMesh).size = Vector2.ONE * 0.2 * dist
			(quads[back].mesh as QuadMesh).size = Vector2.ONE * 0.2 * dist
			quads[front].position = Vector3(0, 0, -dist)
			quads[back].position = Vector3(0, 0, -dist * (1.0 + rel_sep))
			await _wait_frames(3)
			precision.append({"distance_m": dist, "front": "red" if front == 0 else "blue", "seen": _center_color()})
	for q in quads:
		q.queue_free()

	# 3. Occlusion across the near/scaled-space split, with the real scene.
	var sc = main.scene
	for n: Node3D in [sc.earth, sc.moon, sc.craft]:
		n.visible = true
	sc.earth_mat.set_shader_parameter("use_texture", false)
	sc.earth_mat.set_shader_parameter("albedo", Vector3(0, 0, 1))
	sc.earth_mat.set_shader_parameter("ambient", 1.0)
	sc.moon_mat.set_shader_parameter("albedo", Vector3(1, 1, 1))
	sc.moon_mat.set_shader_parameter("ambient", 1.0)
	sc.craft_mat.set_shader_parameter("albedo", Vector3(1, 0, 0))
	sc.craft_mat.set_shader_parameter("ambient", 1.0)
	var f: Dictionary = main.store.sample(0.0)
	var u_moon := D.unit(f["r_moon"])
	var u_sc := D.unit(f["r_sc"])
	var cases := [
		# [name, camera position, look direction, expected centre colour]
		["earth_hides_moon", D.scale(u_moon, -(R_EARTH + 2.0e6)), u_moon, "blue"],
		["craft_hides_earth", D.add(f["r_sc"], D.scale(u_sc, 3.0)), D.scale(u_sc, -1.0), "red"],
		["earth_hides_craft", D.scale(u_sc, -(R_EARTH + 2.0e6)), u_sc, "blue"],
		["moon_visible_ahead", D.sub(f["r_moon"], D.scale(u_moon, 3.0e7)), u_moon, "white"],
	]
	var occlusion := []
	for c: Array in cases:
		sc.update(f, c[1], D.look_rot(c[2], D.v3(0, 0, 1) if absf(c[2][2]) < 0.95 else D.v3(1, 0, 0)))
		await _wait_frames(3)
		occlusion.append({"name": c[0], "expected": c[3], "seen": _center_color(),
			"near": sc.last_near, "far": sc.last_far, "scales": sc.last_scaled})
		main.save_image(_image(), "occlusion_%s.png" % c[0])
	# Restore normal shading for the sweep.
	sc.earth_mat.set_shader_parameter("albedo", Vector3(1, 1, 1))
	sc.earth_mat.set_shader_parameter("ambient", 0.0)
	sc.moon_mat.set_shader_parameter("albedo", Vector3(0.12, 0.12, 0.12))
	sc.moon_mat.set_shader_parameter("ambient", 0.03)
	sc.craft_mat.set_shader_parameter("albedo", Vector3(0.6, 0.6, 0.6))
	sc.craft_mat.set_shader_parameter("ambient", 0.03)
	sc.stars.visible = true

	# 4. Log zoom sweep 0.1 m -> 4e8 m with the full scene; snapshots for review.
	var frames: int = cfg.get("sweep_frames", 240)
	var d0 := log(0.1)
	var d1 := log(4.0e8)
	var snapshots := []
	var next_decade := -1.0
	for i in frames:
		var dist := exp(d0 + (d1 - d0) * i / (frames - 1.0))
		# Look at the spacecraft from the sunward side, slightly above the orbit plane.
		var view_dir := D.unit(D.add(D.unit(D.sub(f["r_sun"], f["r_sc"])), D.v3(0.0, 0.0, 0.3)))
		var cam_pos := D.add(f["r_sc"], D.scale(view_dir, dist + 0.8))
		sc.update(f, cam_pos, D.look_rot(D.scale(view_dir, -1.0), D.v3(0, 0, 1)))
		await RenderingServer.frame_post_draw
		var decade := floorf(log(dist) / log(10.0))
		if decade > next_decade:
			next_decade = decade
			var name := "zoom_1e%d.png" % int(decade)
			main.save_image(_image(), name)
			snapshots.append(name)
	return {"ratio_probe": ratio_probe, "precision": precision, "occlusion": occlusion, "snapshots": snapshots}


# ---------------------------------------------------------------- (b) terminator
func terminator() -> Dictionary:
	var f: Dictionary = main.store.sample(0.0)
	# Numeric: sub-solar point from the viewer's own float64 pipeline.
	var r := D.rot_mat(f["q_itrs_gcrs"])
	var s_itrs := D.unit(D.mat_vec(r, f["r_sun"]))
	var subsolar := {"lon_deg": rad_to_deg(atan2(s_itrs[1], s_itrs[0])), "lat_deg": rad_to_deg(asin(s_itrs[2]))}
	# The float32 Basis given to the Earth node must agree with the float64 math.
	var earth_basis: Basis = D.basis_from(D.transpose(r))
	var sun_godot := D.to_godot(D.unit(f["r_sun"]))
	var via_basis := earth_basis * Vector3(s_itrs[0], s_itrs[1], s_itrs[2])
	var basis_err_deg := rad_to_deg(via_basis.angle_to(sun_godot))

	# Image: view perpendicular to the Sun; the terminator is then a straight
	# line through the disk centre.
	_hide_all()
	main.scene.earth.visible = true
	main.scene.earth_mat.set_shader_parameter("use_texture", false)
	main.scene.earth_mat.set_shader_parameter("ambient", 0.0)
	var sun_dir := D.unit(f["r_sun"])
	var view := D.unit(D.cross(sun_dir, D.v3(0, 0, 1)))
	var dist := 4.0 * R_EARTH
	var cam_rot := D.look_rot(D.scale(view, -1.0), D.v3(0, 0, 1))
	main.scene.update(f, D.scale(view, dist), cam_rot)
	await _wait_frames(4)
	var img := _image()
	img.convert(Image.FORMAT_RGBA8)
	main.save_image(img, "terminator.png")
	var right := PackedFloat64Array([cam_rot[0], cam_rot[3], cam_rot[6]])
	var lit_right := D.dot(sun_dir, right) > 0.0
	var w := img.get_width()
	var h := img.get_height()
	var data := img.get_data()
	var offsets := PackedFloat64Array()
	for y in range(h / 2 - 20, h / 2 + 21):
		var xs := PackedFloat64Array()
		var ls := PackedFloat64Array()
		for k in range(-60, 61):
			var x := w / 2 + k
			var lin := _lin[data[(y * w + x) * 4]]
			if lin > 0.003 and lin < 0.2 and ((k > 0) == lit_right):
				xs.append(x + 0.5)
				ls.append(lin)
		if xs.size() >= 4:
			offsets.append(_zero_crossing(xs, ls) - w / 2.0)
	var fov := deg_to_rad(main.scene.camera.fov)
	var focal := (h / 2.0) / tan(fov / 2.0)
	var disk_px := focal * tan(asin(R_EARTH / dist))
	return {"subsolar": subsolar, "basis_err_deg": basis_err_deg, "offsets_px": offsets,
		"disk_radius_px": disk_px, "t": 0.0}


## x where the least-squares line through (xs, ls) reaches zero.
static func _zero_crossing(xs: PackedFloat64Array, ls: PackedFloat64Array) -> float:
	var n := float(xs.size())
	var mx := 0.0
	var ml := 0.0
	for i in xs.size():
		mx += xs[i] / n
		ml += ls[i] / n
	var sxl := 0.0
	var sxx := 0.0
	for i in xs.size():
		sxl += (xs[i] - mx) * (ls[i] - ml)
		sxx += (xs[i] - mx) * (xs[i] - mx)
	var slope := sxl / sxx
	return mx - ml / slope


# ---------------------------------------------------------------- (b) landmarks
## Camera placed above each landmark using GCRS directions computed in Python;
## the red dot painted at that lat/lon in the test texture must be at the centre.
func landmarks() -> Dictionary:
	var cfg := _cfg("landmarks")
	var img_tex := Image.load_from_file(cfg["texture"])
	main.scene.earth_mat.set_shader_parameter("day_tex", ImageTexture.create_from_image(img_tex))
	main.scene.earth_mat.set_shader_parameter("use_texture", true)
	main.scene.earth_mat.set_shader_parameter("ambient", 1.0)
	_hide_all()
	main.scene.earth.visible = true
	var f: Dictionary = main.store.sample(0.0)
	var dist := 3.0 * R_EARTH
	var out := []
	for lm: Dictionary in cfg["points"]:
		var dir := PackedFloat64Array(lm["dir_gcrs"])
		var up := D.v3(0, 0, 1) if absf(dir[2]) < 0.95 else D.v3(1, 0, 0)
		main.scene.update(f, D.scale(dir, dist), D.look_rot(D.scale(dir, -1.0), up))
		await _wait_frames(3)
		var img := _image()
		img.convert(Image.FORMAT_RGBA8)
		main.save_image(img, "landmark_%s.png" % lm["name"])
		var c := _red_centroid(img)
		out.append({"name": lm["name"], "cx": c.x, "cy": c.y,
			"center": [img.get_width() / 2.0, img.get_height() / 2.0]})
	var fov := deg_to_rad(main.scene.camera.fov)
	var focal: float = (main.scene.viewport.size.y / 2.0) / tan(fov / 2.0)
	return {"points": out, "focal_px": focal, "altitude_m": dist - R_EARTH}


func _red_centroid(img: Image) -> Vector2:
	var data := img.get_data()
	var w := img.get_width()
	var h := img.get_height()
	var sw := 0.0
	var sx := 0.0
	var sy := 0.0
	for y in range(h / 2 - 120, h / 2 + 120):
		for x in range(w / 2 - 120, w / 2 + 120):
			var i := (y * w + x) * 4
			var wgt := _lin[data[i]] - maxf(_lin[data[i + 1]], _lin[data[i + 2]])
			if wgt > 0.05:
				sw += wgt
				sx += wgt * (x + 0.5)
				sy += wgt * (y + 0.5)
	return Vector2(-1, -1) if sw <= 0.0 else Vector2(sx / sw, sy / sw)


# ---------------------------------------------------------------- (c) performance
func perf() -> Dictionary:
	var cfg := _cfg("perf")
	var frames: int = cfg.get("frames", 600)
	var env: Environment = main.scene.env
	env.glow_enabled = true
	env.tonemap_mode = Environment.TONE_MAPPER_AGX
	main.scene.add_model(cfg.get("model_triangles", 100000))
	var rid: RID = main.scene.viewport.get_viewport_rid()
	RenderingServer.viewport_set_measure_render_time(rid, true)
	var gpu := PackedFloat64Array()
	var cpu := PackedFloat64Array()
	var wall := PackedFloat64Array()
	var last := Time.get_ticks_usec()
	for i in frames + 30:
		var u := float(i) / frames
		var f: Dictionary = main.store.sample(u * 60.0)
		# Orbit the spacecraft while zooming 3 m -> 30,000 km and back.
		var dist := exp(log(3.0) + (log(3.0e7) - log(3.0)) * (0.5 - 0.5 * cos(TAU * u)))
		var ang := TAU * u * 1.5
		var dir := D.unit(D.v3(cos(ang), sin(ang), 0.35))
		main.scene.update(f, D.add(f["r_sc"], D.scale(dir, dist)), D.look_rot(D.scale(dir, -1.0), D.v3(0, 0, 1)))
		await RenderingServer.frame_post_draw
		var now := Time.get_ticks_usec()
		if i >= 30:
			gpu.append(RenderingServer.viewport_get_measured_render_time_gpu(rid))
			cpu.append(RenderingServer.viewport_get_measured_render_time_cpu(rid))
			wall.append((now - last) / 1000.0)
		last = now
	var size: Vector2i = main.scene.viewport.size
	return {"gpu_ms": gpu, "render_cpu_ms": cpu, "frame_ms": wall, "res": [size.x, size.y],
		"triangles_model": cfg.get("model_triangles", 100000)}


# ---------------------------------------------------------------- (d) capture
## Deterministic time: frame i shows t = i / fps. Run under --write-movie; the
## runner compares the frames of two runs.
func capture() -> Dictionary:
	var cfg := _cfg("capture")
	var frames: int = cfg.get("frames", 120)
	var fps: float = cfg.get("fps", 60.0)
	var times := PackedFloat64Array()
	for i in frames:
		var t := i / fps
		var f: Dictionary = main.store.sample(t)
		var dir := D.unit(D.v3(1.0, 0.4, 0.3))
		main.scene.update(f, D.add(f["r_sc"], D.scale(dir, 4.0)), D.look_rot(D.scale(dir, -1.0), D.v3(0, 0, 1)))
		times.append(t)
		await RenderingServer.frame_post_draw
	return {"frames": frames, "times": times}


# ---------------------------------------------------------------- (f) HDR headroom
func hdr() -> Dictionary:
	_hide_all()
	var cam: Camera3D = main.scene.camera
	cam.basis = Basis()
	cam.position = Vector3.ZERO
	var exposure := 1.0e-5
	main.scene.env.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	main.scene.env.tonemap_exposure = exposure
	# Unshaded materials ignore EMISSION, so ALBEDO carries the radiance.
	var cases := [["albedo_5e4", 5.0e4, false, -0.6], ["albedo_3e4", 3.0e4, false, 0.0], ["albedo_2e4", 2.0e4, false, 0.6]]
	for c: Array in cases:
		var mi := MeshInstance3D.new()
		var q := QuadMesh.new()
		q.size = Vector2(0.4, 0.4)
		mi.mesh = q
		var mat := ShaderMaterial.new()
		mat.shader = load("res://shaders/flat.gdshader")
		mat.set_shader_parameter("color", Vector3.ONE * c[1])
		mat.set_shader_parameter("as_emission", c[2])
		mi.material_override = mat
		mi.position = Vector3(c[3], 0, -2.0)
		main.scene.viewport.add_child(mi)
	var size: Vector2i = main.scene.viewport.size
	var focal := (size.y / 2.0) / tan(deg_to_rad(cam.fov) / 2.0)
	# Reference frame at exposure 1: the quads must be visible (saturated white).
	main.scene.env.tonemap_exposure = 1.0
	await _wait_frames(4)
	var ref := _image()
	main.save_image(ref, "hdr_ref.png")
	main.scene.env.tonemap_exposure = exposure
	await _wait_frames(4)
	var img := _image()
	main.save_image(img, "hdr.png")
	var out := []
	for c: Array in cases:
		var px := int(size.x / 2.0 + focal * c[3] / 2.0)
		var col := img.get_pixel(px, size.y / 2).srgb_to_linear()
		var col_ref := ref.get_pixel(px, size.y / 2).srgb_to_linear()
		out.append({"name": c[0], "value": c[1], "expected_linear": c[1] * exposure, "seen_linear": col.r,
			"seen_linear_at_exposure_1": col_ref.r})
	return {"exposure": exposure, "cases": out}


# ---------------------------------------------------------------- (e) transport
## Count live rows and gaps in t_s until the session ends.
func transport() -> Dictionary:
	var rows := 0
	var gaps := 0
	var last_t := -1.0
	var t0 := Time.get_ticks_usec()
	var deadline := Time.get_ticks_msec() + 120000
	var t_index: int = main.stream.names.find("t_s")
	while not main.stream.session_ended and Time.get_ticks_msec() < deadline:
		for m: Array in main.stream.poll_messages():
			if m[0] == main.stream.Msg.ROW:
				var r: Array = main.stream.decode_row(m[2])
				var tv: float = (r[1] as PackedFloat64Array)[t_index]
				if last_t >= 0.0 and absf(tv - last_t - main.config.get("transport", {}).get("dt", 0.001)) > 1e-9:
					gaps += 1
				last_t = tv
				rows += 1
		OS.delay_usec(200)
	return {"rows": rows, "gaps": gaps, "last_t": last_t, "elapsed_s": (Time.get_ticks_usec() - t0) / 1e6,
		"mode": main.stream.hello.get("mode", "")}
