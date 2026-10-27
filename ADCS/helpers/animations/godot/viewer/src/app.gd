extends Node
## Generalized ADCS viewer.
##
##   <viewer> -- --connect HOST:PORT --token T [--assets DIR]     (from python -m ADCS.viz view)
##   <viewer> -- --log RUN.vizlog [--run ID] [--assets DIR]       (read a vizlog from disk)
##
## Options: --res WxH (fixed render size), --mode orbit|earth, --frame lvlh|inertial|body,
## --rate R, --max-face N (downscale textures), --check NAME --config FILE --out DIR,
## --render JOB.json --out DIR (offline video frames, see src/render.gd).

const D := preload("res://src/core/dmath.gd")
const Clock := preload("res://src/core/viz_clock.gd")
const StreamSource := preload("res://src/io/stream_source.gd")
const FileSource := preload("res://src/io/file_source.gd")
const Assets := preload("res://src/scene/assets.gd")
const World := preload("res://src/scene/world.gd")
const CameraRig := preload("res://src/camera/camera_rig.gd")
const Hud := preload("res://src/ui/hud.gd")
const Checks := preload("res://src/checks.gd")
const Renderer := preload("res://src/render.gd")

var opts: Dictionary = {}
var source: RefCounted  ## StreamSource or FileSource
var clock := Clock.new()
var assets := Assets.new()
var world: World
var rig := CameraRig.new()
var hud: Hud
var loaded_assets: Dictionary = {}
var last_frame: Dictionary = {}
var _fps_t := 0.0
var checks: RefCounted
var renderer: RefCounted
var focus_menu: PopupMenu
var pending_focus := ""
var pointer_down := Vector2.ZERO
var asset_prompt_file := ""
var asset_ready_file := ""
var asset_prompt_finished := false


func _ready() -> void:
	opts = parse_args(OS.get_cmdline_user_args())
	var size := Vector2i.ZERO
	if opts.has("res"):
		var wh := String(opts["res"]).split("x")
		size = Vector2i(int(wh[0]), int(wh[1]))
	world = World.new(self, size)
	hud = Hud.new()
	add_child(hud)
	hud.setup_settings(world)
	asset_prompt_file = String(opts.get("asset-prompt-file", ""))
	asset_ready_file = String(opts.get("asset-ready-file", ""))
	if asset_prompt_file != "" and not FileAccess.file_exists(asset_prompt_file):
		hud.show_asset_notice(true)
	focus_menu = PopupMenu.new()
	focus_menu.add_item("Focus on Object", 0)
	focus_menu.id_pressed.connect(func(_id: int) -> void: _focus(pending_focus))
	add_child(focus_menu)
	get_viewport().size_changed.connect(func() -> void: world.resize_to(get_viewport().get_visible_rect().size))

	_log("opening data source")
	if not _open_source():
		var msg: String = source.error_message if source != null else "nothing to show"
		if opts.has("check"):
			Checks.finish(self, {"error": msg})
		elif opts.has("render"):
			Renderer.new(self)._finish(String(opts.get("out", OS.get_user_data_dir())), {"error": msg})
		else:
			hud.status.text = "Cannot open data: %s\n\nRun from Python: python -m ADCS.viz view run.vizlog" % msg
			set_process(false)
		_signal_asset_ready()
		return

	_log("data: %d rows, %d assets" % [source.store.n_rows, source.assets.size()])
	assets.max_face = int(opts.get("max-face", 0))
	if assets.open(String(opts.get("assets", ""))):
		loaded_assets = world.load_assets(assets)
	else:
		world.load_assets(assets)  # flat colours, procedural stars
	world.solar.load_textures(String(opts.get("assets", "")), world.earth_mat, world.moon_mat, world.sun.material_override)
	_log("textures loaded: %s" % [assets.load_ms])
	world.craft.build(source.scene, source.assets)
	_log("spacecraft: %s, %d levels of detail, radius %.3f m" % [world.craft.source, world.craft.lods.size(), world.craft.radius])
	rig.fit_to_craft(world.craft.radius)
	rig.frame_kind = String(opts.get("frame", "lvlh"))
	if String(opts.get("mode", "orbit")) == "earth":
		rig.mode = CameraRig.Mode.EARTH
	world.set_exposure(String(opts.get("exposure", "artistic")), float(opts.get("ev", 0.0)))
	world.set_quality(String(opts.get("quality", "fast")))
	var scene: Dictionary = source.scene.duplicate()
	if not scene.has("overlays"):  # always offer body axes (off until toggled)
		scene["overlays"] = [{"id": "body_axes", "type": "triad", "frame": "body", "visible": false}]
	world.overlays.build(scene, source.manifest, source.store, world.craft, world.viewport)
	world.craft.set_joint_store(source.store)
	hud.layers.setup(world.overlays)
	_log("overlays: %d" % world.overlays.nodes.size())

	_update_range()
	clock.t = clock.t_min
	clock.rate = float(opts.get("rate", 1.0 if clock.t_max - clock.t_min < 3600.0 else 60.0))
	clock.follow_live = source.live
	hud.setup(clock, source)
	hud.timeline.reset_camera_requested.connect(reset_camera)

	_log("ready")
	_signal_asset_ready()
	if opts.has("render"):
		set_process(false)
		renderer = Renderer.new(self)  # keep a reference (see checks below)
		renderer.run(String(opts["render"]))
		return
	if opts.has("check"):
		set_process(false)
		# Keep a reference: a RefCounted freed mid-await silently drops its coroutine.
		checks = Checks.new(self)
		checks.run(String(opts["check"]))


func _signal_asset_ready() -> void:
	if asset_ready_file == "":
		return
	var file := FileAccess.open(asset_ready_file, FileAccess.WRITE)
	if file != null:
		file.close()


func _open_source() -> bool:
	if opts.has("connect"):
		var s := StreamSource.new()
		source = s
		var hp := String(opts["connect"]).rsplit(":", true, 1)
		if not s.open(hp[0], int(hp[1]), String(opts.get("token", "")), String(opts.get("run", ""))):
			return false
		if not s.live:
			return s.receive_all(120.0)
		var deadline := Time.get_ticks_msec() + 30000
		while s.store.n_rows < 2 and Time.get_ticks_msec() < deadline:
			s.poll()
			OS.delay_msec(5)
		return s.store.n_rows >= 2
	if opts.has("log"):
		var f := FileSource.new()
		source = f
		return f.open(String(opts["log"]), String(opts.get("run", "")))
	return false


func _update_range() -> void:
	clock.set_range(source.store.t_min(), source.store.t_max())


var _frame_count := 0


func _process(delta: float) -> void:
	if not asset_prompt_finished and asset_prompt_file != "" and FileAccess.file_exists(asset_prompt_file):
		asset_prompt_finished = true
		hud.show_asset_notice(false)
		world.solar.load_textures(String(opts.get("assets", "")), world.earth_mat, world.moon_mat, world.sun.material_override)
	_frame_count += 1
	if _frame_count == 1:
		_log("first frame")
	var new_rows: bool = source.poll()
	if source.live and new_rows:
		_update_range()
	clock.advance(delta)
	if source.has_method("request_ephemeris"):
		source.request_ephemeris(clock.t)
	render_at(clock.t)
	_fps_t += delta
	hud.refresh(clock, last_frame, _status_lines())


## Pose the world for time t with the current camera. Returns the frame sample.
func render_at(t: float) -> Dictionary:
	last_frame = source.store.sample_extended(t)
	if source.has_method("apply_ephemeris"):
		source.apply_ephemeris(last_frame, t)
	var cam: Array = rig.compute(last_frame)
	world.update(last_frame, cam[0], cam[1])
	return last_frame


func _status_lines() -> PackedStringArray:
	var f := last_frame
	var lines := PackedStringArray()  # the time itself is on the timeline
	var alt := (D.norm(f["r_sc"]) - World.R_EARTH) / 1000.0
	var speed := D.norm(f["v_sc"]) / 1000.0
	lines.append("Altitude %.1f km   speed %.3f km/s" % [alt, speed])
	var sf: float = f["shadow_factor"]
	lines.append("Sunlit" if sf >= 1.0 else ("Eclipse" if sf <= 0.0 else "Penumbra %.0f%%" % (100.0 * sf)))
	if f.has("pointing_err"):
		lines.append("Pointing error %.3f deg" % rad_to_deg((f["pointing_err"] as PackedFloat64Array)[0]))
	lines.append(rig.describe())
	var live := "   LIVE" if source.live and clock.follow_live else ""
	lines.append("%d fps   model: %s%s" % [Engine.get_frames_per_second(), world.craft.source, live])
	return lines


## UTC for t seconds after the epoch (ignores a leap second inside the run).
func utc_string(t: float) -> String:
	var iso: Variant = source.manifest.get("epoch", {}).get("iso_utc")
	if not iso is String:
		return "t = %.1f s" % t
	var base := Time.get_unix_time_from_datetime_string(String(iso).trim_suffix("Z"))
	var frac := float(String(iso).get_slice(".", 1).trim_suffix("Z")) / 1000.0 if "." in String(iso) else 0.0
	var u := base + frac + t
	return "%s UTC   (t = %.1f s)" % [Time.get_datetime_string_from_unix_time(int(floor(u)), true), t]


func _unhandled_input(event: InputEvent) -> void:
	var mouse := event as InputEventMouseButton
	if mouse != null and mouse.button_index == MOUSE_BUTTON_LEFT:
		if mouse.pressed:
			pointer_down = mouse.position
		elif mouse.position.distance_to(pointer_down) < 5.0 and not last_frame.is_empty():
			pending_focus = world.pick_body(mouse.position, last_frame)
			if pending_focus != "":
				focus_menu.position = Vector2i(mouse.position)
				focus_menu.popup()
	if rig.handle_input(event):
		return
	var key := event as InputEventKey
	if key != null and key.pressed and key.keycode in [KEY_LEFT, KEY_RIGHT]:  # repeats while held
		var d := -1 if key.keycode == KEY_LEFT else 1
		if key.shift_pressed:
			hud.timeline.jump_event(d)
		else:
			hud.timeline.step_row(d)
		return
	if key != null and key.pressed and not key.echo:
		match key.keycode:
			KEY_B: hud.timeline.add_bookmark()
			KEY_T: hud.timeline.open_time_entry()
			KEY_F: hud.timeline.fit_view()
			KEY_I: hud.card.enabled = not hud.card.enabled
			KEY_SPACE: clock.toggle_pause()
			KEY_R:
				clock.seek(clock.t_min)
				clock.paused = true
			KEY_PERIOD: clock.faster()
			KEY_COMMA: clock.slower()
			KEY_1: _orbit("lvlh")
			KEY_2: _orbit("inertial")
			KEY_3: _orbit("body")
			KEY_4: rig.mode = CameraRig.Mode.EARTH
			KEY_L: clock.follow_live = source.live
			KEY_V: world.overlays.toggle_group("vectors")
			KEY_A: world.overlays.toggle_group("axes")
			KEY_G: world.overlays.toggle_group("tracks")
			KEY_O: world.overlays.toggle_all()
			KEY_H: hud.help.visible = not hud.help.visible
			KEY_E: world.set_exposure("physical" if world.exposure_mode == "artistic" else "artistic", world.exposure_ev)
			KEY_Q:
				var names := World.QUALITY.keys()
				world.set_quality(names[(names.find(world.quality) + 1) % names.size()])
			KEY_BRACKETLEFT: world.set_exposure(world.exposure_mode, world.exposure_ev - 1.0)
			KEY_BRACKETRIGHT: world.set_exposure(world.exposure_mode, world.exposure_ev + 1.0)
			KEY_HOME:
				reset_camera()


func reset_camera() -> void:
	_orbit("lvlh")
	rig.fit_to_craft(world.craft.radius)
	rig.yaw = deg_to_rad(-140.0)
	rig.pitch = deg_to_rad(18.0)


func _orbit(kind: String) -> void:
	rig.mode = CameraRig.Mode.ORBIT
	rig.focus_name = "spacecraft"
	hud.focus.text = "FOCUS: SPACECRAFT"
	rig.frame_kind = kind


func _focus(name: String) -> void:
	if name == "spacecraft":
		_orbit("lvlh")
		return
	rig.focus_name = name
	rig.mode = CameraRig.Mode.EARTH if name == "earth" else CameraRig.Mode.BODY
	var radii := {"sun": 695700000.0, "earth": World.R_EARTH, "moon": World.R_MOON,
		"mercury": 2439500.0, "venus": 6052000.0, "mars": 3396000.0,
		"jupiter": 71492000.0, "saturn": 60268000.0,
		"uranus": 25559000.0, "neptune": 24764000.0}
	rig.earth_dist = 4.0 * radii.get(name, World.R_EARTH)
	rig.focus_radius = radii.get(name, World.R_EARTH)
	hud.focus.text = "FOCUS: " + name.to_upper()


func _log(msg: String) -> void:
	print("[viewer %.1fs] %s" % [Time.get_ticks_msec() / 1000.0, msg])


static func parse_args(argv: PackedStringArray) -> Dictionary:
	var out := {}
	var i := 0
	while i < argv.size():
		var a := argv[i]
		if a.begins_with("--"):
			var key := a.substr(2)
			if i + 1 < argv.size() and not argv[i + 1].begins_with("--"):
				out[key] = argv[i + 1]
				i += 1
			else:
				out[key] = true
		i += 1
	return out
