extends RefCounted
## Atmosphere tables from `python -m ADCS.viz.assets build atmosphere`
## (Hillaire 2020; transmittance parameterised as in Bruneton 2017).
##
## The GPU gets both tables as textures (shaders/atmosphere.gdshaderinc).
## Transmittance is also kept on the CPU to colour the sunlight that reaches
## the spacecraft: reddened when the Sun is seen through the limb near eclipse.

const D := preload("res://src/core/dmath.gd")

var params: Dictionary = {}
var transmittance_tex: ImageTexture
var multiscatter_tex: ImageTexture
var _t: PackedFloat32Array
var _tw := 0
var _th := 0
var bottom := 6378137.0
var top := 6458137.0


func load_from(dir: String, manifest: Dictionary) -> bool:
	var p: Variant = JSON.parse_string(FileAccess.get_file_as_string(dir.path_join("atmosphere.json")))
	if not p is Dictionary:
		return false
	params = p
	# Older cached LUTs encode a shorter/thinner profile. Stretching them would
	# break their transmittance mapping, so use the matching procedural profile
	# until the atmosphere product is rebuilt.
	if absf(float(params.get("top_radius", 0.0)) - 6458137.0) > 1.0 \
			or float(params.get("rayleigh_scale_height", 0.0)) < 10500.0 \
			or float(params.get("mie_scale_height", 0.0)) < 1450.0:
		bottom = 6378137.0
		top = 6458137.0
		return false
	bottom = float(params["bottom_radius"])
	top = float(params["top_radius"])
	for f: Dictionary in manifest.get("files", []):
		if not f.has("shape"):
			continue
		var shape: Array = f["shape"]
		var bytes := FileAccess.get_file_as_bytes(dir.path_join(f["name"]))
		var img := Image.create_from_data(int(shape[1]), int(shape[0]), false, Image.FORMAT_RGBF, bytes)
		var tex := ImageTexture.create_from_image(img)
		if String(f["name"]).begins_with("transmittance"):
			transmittance_tex = tex
			_t = bytes.to_float32_array()
			_tw = int(shape[1])
			_th = int(shape[0])
		else:
			multiscatter_tex = tex
	return transmittance_tex != null and multiscatter_tex != null


func apply(mat: ShaderMaterial) -> void:
	mat.set_shader_parameter("has_atmosphere", true)
	mat.set_shader_parameter("use_atmosphere_lut", transmittance_tex != null and multiscatter_tex != null)
	mat.set_shader_parameter("atm_bottom", bottom)
	mat.set_shader_parameter("atm_top", top)
	if transmittance_tex == null or multiscatter_tex == null:
		return
	var vec := func(k: String) -> Vector3:
		var a: Array = params[k]
		return Vector3(a[0], a[1], a[2])
	mat.set_shader_parameter("rayleigh_scattering", vec.call("rayleigh_scattering"))
	mat.set_shader_parameter("rayleigh_h", float(params["rayleigh_scale_height"]))
	mat.set_shader_parameter("mie_scattering", vec.call("mie_scattering"))
	mat.set_shader_parameter("mie_extinction", vec.call("mie_extinction"))
	mat.set_shader_parameter("mie_h", float(params["mie_scale_height"]))
	mat.set_shader_parameter("mie_g", float(params["mie_g"]))
	mat.set_shader_parameter("ozone_absorption", vec.call("ozone_absorption"))
	mat.set_shader_parameter("ozone_center", float(params["ozone_center"]))
	mat.set_shader_parameter("ozone_half_width", float(params["ozone_half_width"]))
	mat.set_shader_parameter("transmittance_lut", transmittance_tex)
	mat.set_shader_parameter("multiscatter_lut", multiscatter_tex)


## Transmittance from radius r (<= top) to the top along zenith cosine mu (CPU, bilinear).
func transmittance(r: float, mu: float) -> Vector3:
	if _t.is_empty():
		var h := maxf(r - bottom, 0.0)
		var closest := r * sqrt(maxf(1.0 - mu * mu, 0.0))
		var h_along := maxf((closest if mu < 0.0 else r) - bottom, 0.0)
		var rayleigh_depth := 10500.0 * exp(-h_along / 10500.0) / sqrt(mu * mu + 10500.0 / bottom)
		var mie_depth := 1450.0 * exp(-h_along / 1450.0) / sqrt(mu * mu + 1450.0 / bottom)
		if mu < 0.0:
			rayleigh_depth *= 2.0
			mie_depth *= 2.0
		var ozone_fraction := clampf((40000.0 - h) / 30000.0, 0.0, 1.0)
		var ozone_depth := 15000.0 * ozone_fraction / sqrt(mu * mu + 15000.0 / bottom)
		return Vector3(
			exp(-(5.802e-6 * rayleigh_depth + 4.44e-6 * mie_depth + 0.65e-6 * ozone_depth)),
			exp(-(13.558e-6 * rayleigh_depth + 4.44e-6 * mie_depth + 1.881e-6 * ozone_depth)),
			exp(-(33.1e-6 * rayleigh_depth + 4.44e-6 * mie_depth + 0.085e-6 * ozone_depth)))
	var H := sqrt(top * top - bottom * bottom)
	var rho := sqrt(maxf(r * r - bottom * bottom, 0.0))
	var d := maxf(-r * mu + sqrt(maxf(r * r * (mu * mu - 1.0) + top * top, 0.0)), 0.0)
	var d_min := top - r
	var d_max := rho + H
	var x_mu := (d - d_min) / maxf(d_max - d_min, 1e-9)
	var x := clampf((0.5 / _tw + x_mu * (1.0 - 1.0 / _tw)) * _tw - 0.5, 0.0, _tw - 1.0)
	var y := clampf((0.5 / _th + rho / H * (1.0 - 1.0 / _th)) * _th - 0.5, 0.0, _th - 1.0)
	var x0 := int(x)
	var y0 := int(y)
	var x1 := mini(x0 + 1, _tw - 1)
	var y1 := mini(y0 + 1, _th - 1)
	var fx := x - x0
	var fy := y - y0
	return (_px(x0, y0) * (1 - fx) + _px(x1, y0) * fx) * (1 - fy) + (_px(x0, y1) * (1 - fx) + _px(x1, y1) * fx) * fy


func _px(x: int, y: int) -> Vector3:
	var i := (y * _tw + x) * 3
	return Vector3(_t[i], _t[i + 1], _t[i + 2])


## Colour of sunlight arriving at GCRS position p (m, Earth-centred) from direction s:
## the transmittance of the ray's chord through the atmosphere, (0,0,0) if it hits the ground.
func sunlight(p: PackedFloat64Array, s: PackedFloat64Array) -> Vector3:
	var r := D.norm(p)
	var mu := D.dot(p, s) / r
	if r <= top:
		if mu < -sqrt(maxf(1.0 - (bottom / r) * (bottom / r), 0.0)):
			return Vector3.ZERO
		return transmittance(r, mu)
	# Above the atmosphere: does the ray toward the Sun cross it?
	var b := -D.dot(p, s)  # distance to closest approach (negative: Sun side)
	var closest := sqrt(maxf(r * r - b * b, 0.0))
	if mu >= 0.0 or closest >= top:
		return Vector3.ONE
	if closest <= bottom:
		return Vector3.ZERO
	# Entry point on the top sphere, then the transmittance of the whole chord.
	var t_in := b - sqrt(top * top - closest * closest)
	var entry := D.add(p, D.scale(s, t_in))
	return transmittance(top, D.dot(entry, s) / top)
