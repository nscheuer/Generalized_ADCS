extends RefCounted
## Offline rendering: `--render JOB.json --out DIR` (started by ADCS.viz.render).
##
## For every sub-frame listed in the job ({i, j, t, cam}) the camera and the
## simulation time are set explicitly, one frame is drawn and the image is
## saved as DIR/<i>_<j>.png. Nothing depends on the wall clock, so a job gives
## the same images every time on the same machine. Supersampling and motion
## blur are combined afterwards in Python (linear light); motion-blur sub-frames
## are drawn without overlays and HUD (layer "scene"), which are laid over sharp.

const CameraRig := preload("res://src/camera/camera_rig.gd")
const WARMUP_FRAMES := 8  ## let shaders compile and textures upload before the first saved frame

var app  # app.gd


func _init(a: Variant) -> void:
	app = a


func run(job_path: String) -> void:
	await app.get_tree().process_frame
	var out_dir := String(app.opts.get("out", OS.get_user_data_dir()))
	DirAccess.make_dir_recursive_absolute(out_dir)
	var job: Variant = JSON.parse_string(FileAccess.get_file_as_string(job_path))
	if not job is Dictionary or not (job as Dictionary).has("samples"):
		_finish(out_dir, {"error": "cannot read render job %s" % job_path})
		return
	var w: RefCounted = app.world
	w.set_quality(String(job.get("quality", "high")))
	w.set_exposure(String(job.get("exposure", "artistic")), float(job.get("ev", 0.0)))
	for g: String in (job.get("overlays", {}) as Dictionary):
		w.overlays.set_group(g, bool(job["overlays"][g]))
	var ui := bool(job.get("ui", false))
	app.hud.visible = ui
	app.hud.status.visible = false  # frame rate and similar have no meaning in a video
	if ui:
		DisplayServer.window_set_size(w.viewport.size)
	var clock: RefCounted = app.clock
	clock.follow_live = false
	clock.paused = false  # drawn as playing: no state card
	clock.rate = float(job.get("playback_rate", 1.0))  # what the speed pill shows
	var samples: Array = job["samples"]
	var t0 := Time.get_ticks_usec()
	_apply(samples[0])
	for k in WARMUP_FRAMES:
		_draw(float(samples[0]["t"]), ui)
		await RenderingServer.frame_post_draw
	for s: Dictionary in samples:
		_apply(s)
		_draw(float(s["t"]), ui, String(s.get("layer", "all")))
		await RenderingServer.frame_post_draw
		var tex: Texture2D = app.get_viewport().get_texture() if ui else w.viewport.get_texture()
		var img := tex.get_image()
		img.convert(Image.FORMAT_RGB8)
		var err := img.save_png(out_dir.path_join("%06d_%02d.png" % [int(s["i"]), int(s["j"])]))
		if err != OK:
			_finish(out_dir, {"error": "cannot write frames to %s (error %d)" % [out_dir, err]})
			return
	var wall := (Time.get_ticks_usec() - t0) / 1.0e6
	_finish(out_dir, {"sub_frames": samples.size(), "size": [w.viewport.size.x, w.viewport.size.y],
		"quality": w.quality, "exposure": w.exposure_mode, "wall_s": wall,
		"ms_per_sub_frame": 1000.0 * wall / maxf(samples.size(), 1)})


func _apply(s: Dictionary) -> void:
	var cam: Dictionary = s["cam"]
	var rig: CameraRig = app.rig
	if String(cam.get("mode", "orbit")) == "earth":
		rig.mode = CameraRig.Mode.EARTH
		rig.earth_dist = float(cam["dist_m"])
	else:
		rig.mode = CameraRig.Mode.ORBIT
		rig.frame_kind = String(cam.get("frame", "lvlh"))
		rig.dist = float(cam["dist_m"])
	rig.yaw = deg_to_rad(float(cam.get("yaw_deg", -140.0)))
	rig.pitch = deg_to_rad(float(cam.get("pitch_deg", 18.0)))
	app.world.fov_deg = float(cam.get("fov_deg", 45.0))


## layer "all": everything; "scene": without overlays and HUD (motion-blur sub-frames).
func _draw(t: float, ui: bool, layer: String = "all") -> void:
	var all := layer != "scene"
	app.world.overlays.enabled = all
	for n: Node3D in app.world.overlays.nodes:
		n.visible = all and bool(app.world.overlays.shown.get(app.world.overlays.group_of(n), true))
	app.hud.visible = ui and all
	app.clock.t = t
	app.render_at(t)
	if ui and all:
		app.hud.refresh(app.clock, app.last_frame, PackedStringArray())


func _finish(out_dir: String, result: Dictionary) -> void:
	result["adapter"] = RenderingServer.get_video_adapter_name()
	result["adapter_vendor"] = RenderingServer.get_video_adapter_vendor()
	result["driver"] = RenderingServer.get_video_adapter_api_version()
	result["renderer"] = ProjectSettings.get_setting("rendering/renderer/rendering_method", "forward_plus")
	result["godot"] = Engine.get_version_info()["string"]
	var version := str(ProjectSettings.get_setting("application/config/version", ""))
	result["viewer_version"] = version if version != "" else "dev"
	var f := FileAccess.open(out_dir.path_join("render.json"), FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(result))
		f.close()
	if result.has("error"):
		printerr("render error: ", result["error"])
	app.get_tree().quit(1 if result.has("error") else 0)
