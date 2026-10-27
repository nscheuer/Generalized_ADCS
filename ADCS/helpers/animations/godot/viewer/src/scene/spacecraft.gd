extends Node3D
## The spacecraft: an imported model (glTF levels of detail in BODY metres, from
## ADCS.viz.mesh_import) or, without one, boxes built from the drag/SRP panels.
## Local axes are the body axes.

const D := preload("res://src/core/dmath.gd")

var radius := 0.5  ## bounding radius [m], drives the frustum and LOD choice
var lod := 0
var lods: Array[Node3D] = []
var source := "none"
var average_color := Color(0.72, 0.72, 0.75)
var wheel_nodes: Dictionary = {}  ## [rw index, lod id] -> [Node3D, axis, rest Transform3D, pivot]
var _wheel_angle: Dictionary = {}
var hinge_nodes: Array = []  ## [Node3D, axis, pivot, rest Transform3D, channel]
var _joint_store: RefCounted
var _last_t := NAN
var _end_materials: Array = []  ## [MeshInstance3D, surface, original override, faded material]
var _after_simulation := false


## Build from the scene description and the model files received as assets.
func build(scene: Dictionary, assets: Dictionary) -> void:
	for c in get_children():
		c.queue_free()
	lods.clear()
	_end_materials.clear()
	_after_simulation = false
	var mesh: Variant = scene.get("mesh")
	if mesh is Dictionary and not assets.is_empty():
		var names := assets.keys()
		names.sort()  # <name>_lod0.glb, _lod1, ...
		for n: String in names:
			var entry: Dictionary = assets[n]
			if String(entry.get("meta", {}).get("kind", "")) != "model_lod":
				continue
			var node := _load_glb(entry["data"])
			if node != null:
				node.name = n.get_basename()
				add_child(node)
				lods.append(node)
	if not lods.is_empty():
		source = "model"
	else:
		var fallback := _panel_boxes(scene)
		add_child(fallback)
		lods.append(fallback)
		source = "panels" if not (scene.get("panels", []) as Array).is_empty() else "box"
	radius = maxf(_bounding_radius(lods[0]), 0.05)
	average_color = _average_model_color(lods[0]) if source == "model" else Color(0.72, 0.72, 0.75)
	_cache_end_materials()
	_find_wheels(scene)
	_find_hinges(scene)
	choose_lod(1e9)


func _load_glb(data: PackedByteArray) -> Node3D:
	var doc := GLTFDocument.new()
	var state := GLTFState.new()
	if doc.append_from_buffer(data, "", state) != OK:
		push_error("could not parse spacecraft model")
		return null
	var root := doc.generate_scene(state) as Node3D
	if root != null:
		_apply_vertex_colors(root)
	return root


## glTF says COLOR_0 multiplies the base colour, but Godot's default material
## ignores it: give coloured surfaces a material that uses it.
static func _apply_vertex_colors(node: Node) -> void:
	for mi in node.find_children("*", "MeshInstance3D", true, false):
		var m := (mi as MeshInstance3D).mesh
		if m == null:
			continue
		for surf in m.get_surface_count():
			if m.surface_get_format(surf) & Mesh.ARRAY_FORMAT_COLOR == 0:
				continue
			var base := m.surface_get_material(surf) as StandardMaterial3D
			var mat := base.duplicate() as StandardMaterial3D if base != null else StandardMaterial3D.new()
			mat.vertex_color_use_as_albedo = true
			if base == null:
				mat.metallic = 0.2
				mat.roughness = 0.6
			(mi as MeshInstance3D).set_surface_override_material(surf, mat)


static func _average_model_color(root: Node3D) -> Color:
	var total := Color(0, 0, 0, 0)
	var count := 0
	for child in [root] + root.find_children("*", "MeshInstance3D", true, false):
		if not child is MeshInstance3D or child.mesh == null:
			continue
		var mesh := child as MeshInstance3D
		for surface in mesh.mesh.get_surface_count():
			var material := mesh.get_active_material(surface)
			if not material is StandardMaterial3D:
				continue
			var standard := material as StandardMaterial3D
			var color := standard.albedo_color
			if standard.vertex_color_use_as_albedo:
				var arrays := mesh.mesh.surface_get_arrays(surface)
				var vertex_colors: PackedColorArray = arrays[Mesh.ARRAY_COLOR]
				if not vertex_colors.is_empty():
					var vertex_average := Color(0, 0, 0, 0)
					for vertex_color in vertex_colors:
						vertex_average += vertex_color
					color *= vertex_average / float(vertex_colors.size())
			if standard.albedo_texture != null:
				var image := standard.albedo_texture.get_image()
				if image != null and not image.is_empty():
					image.resize(1, 1, Image.INTERPOLATE_LANCZOS)
					color *= image.get_pixel(0, 0)
			total += color
			count += 1
	return total / float(count) if count > 0 else Color(0.72, 0.72, 0.75)


func _cache_end_materials() -> void:
	if source != "model":
		return
	for lod_node in lods:
		for child in [lod_node] + lod_node.find_children("*", "MeshInstance3D", true, false):
			if not child is MeshInstance3D or child.mesh == null:
				continue
			var mi := child as MeshInstance3D
			for surf in mi.mesh.get_surface_count():
				var original: Material = mi.get_surface_override_material(surf)
				var active := mi.get_active_material(surf)
				var faded := active.duplicate() as StandardMaterial3D if active is StandardMaterial3D else StandardMaterial3D.new()
				faded.albedo_color = Color(0.65, 0.65, 0.65, 0.8)
				faded.albedo_texture = null
				faded.vertex_color_use_as_albedo = false
				faded.emission_enabled = false
				faded.detail_enabled = false
				faded.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
				_end_materials.append([mi, surf, original, faded])


func set_after_simulation(active: bool) -> void:
	if active == _after_simulation:
		return
	_after_simulation = active
	for entry: Array in _end_materials:
		(entry[0] as MeshInstance3D).set_surface_override_material(entry[1], entry[3] if active else entry[2])


func _panel_boxes(scene: Dictionary) -> Node3D:
	var root := Node3D.new()
	root.name = "panels"
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.72, 0.72, 0.75)
	mat.metallic = 0.4
	mat.roughness = 0.45
	var panels: Array = scene.get("panels", [])
	if panels.is_empty():
		# No geometry at all: a cube sized from the inertia (sqrt(2 tr J / m)).
		var sc: Dictionary = scene.get("spacecraft", {})
		var side := 0.3
		if sc.has("inertia_com_kgm2") and sc.has("mass_kg") and float(sc["mass_kg"]) > 0.0:
			var j: Array = sc["inertia_com_kgm2"]
			side = sqrt(2.0 * (float(j[0][0]) + float(j[1][1]) + float(j[2][2])) / float(sc["mass_kg"]))
		var box := MeshInstance3D.new()
		var bm := BoxMesh.new()
		bm.size = Vector3.ONE * side
		box.mesh = bm
		box.material_override = mat
		root.add_child(box)
		return root
	for p: Dictionary in panels:
		var c: Array = p["center_body_m"]
		var nrm: Array = p["normal_body"]
		var side := sqrt(float(p["area_m2"]))
		var box := MeshInstance3D.new()
		var bm := BoxMesh.new()
		bm.size = Vector3(side, side, 0.004)
		box.mesh = bm
		box.material_override = mat
		var n := Vector3(nrm[0], nrm[1], nrm[2]).normalized()
		var up := Vector3.UP if absf(n.y) < 0.9 else Vector3.RIGHT
		var x := up.cross(n).normalized()
		box.transform = Transform3D(Basis(x, n.cross(x), n), Vector3(c[0], c[1], c[2]) - n * 0.002)
		root.add_child(box)
	return root


## Largest distance from the body origin of any mesh corner under `node`.
static func _bounding_radius(node: Node, xform: Transform3D = Transform3D.IDENTITY) -> float:
	var t := xform * (node as Node3D).transform if node is Node3D else xform
	var r := 0.0
	if node is MeshInstance3D:
		var aabb := (node as MeshInstance3D).get_aabb()
		for i in 8:
			r = maxf(r, (t * aabb.get_endpoint(i)).length())
	for c in node.get_children():
		r = maxf(r, _bounding_radius(c, t))
	return r


## Centre of the meshes under `node`, in the node's parent coordinates.
static func _mesh_center(node: Node3D) -> Vector3:
	var acc := AABB()
	var first := true
	for mi in [node] + node.find_children("*", "MeshInstance3D", true, false):
		if mi is MeshInstance3D:
			var rel := node.transform * _relative_transform(mi as Node3D, node)
			var box := rel * (mi as MeshInstance3D).get_aabb()
			acc = box if first else acc.merge(box)
			first = false
	return acc.get_center()


static func _relative_transform(n: Node3D, ancestor: Node3D) -> Transform3D:
	var t := Transform3D.IDENTITY
	var cur: Node = n
	while cur != null and cur != ancestor:
		if cur is Node3D:
			t = (cur as Node3D).transform * t
		cur = cur.get_parent()
	return t


func _find_wheels(scene: Dictionary) -> void:
	wheel_nodes.clear()
	if source != "model":
		return
	var rw_index := 0
	for a: Dictionary in scene.get("actuators", []):
		if a.get("type", "") != "rw":
			continue
		var node_name: Variant = a.get("node")
		if node_name is String:
			for l in lods:
				var n := l.find_child(node_name, true, false) as Node3D
				if n != null:
					var ax: Array = a["axis_body"]
					wheel_nodes[[rw_index, l.get_instance_id()]] = [n, Vector3(ax[0], ax[1], ax[2]).normalized(),
						n.transform, _mesh_center(n)]
		rw_index += 1


## Hinged parts: model nodes with articulation "hinge", driven by the channel
## "joint_<node>" (rad) when the log has it. Pivot: "pivot_body" from the model
## config, else the corner of the part nearest the body origin.
func _find_hinges(scene: Dictionary) -> void:
	hinge_nodes.clear()
	var mesh: Variant = scene.get("mesh")
	if source != "model" or not mesh is Dictionary:
		return
	for node_name: String in (mesh.get("nodes", {}) as Dictionary):
		var entry: Dictionary = mesh["nodes"][node_name]
		if entry.get("articulation") != "hinge" or not entry.has("axis_body"):
			continue
		var ax: Array = entry["axis_body"]
		for l in lods:
			var n := l.find_child(node_name, true, false) as Node3D
			if n == null:
				continue
			var pivot: Vector3
			if entry.has("pivot_body"):
				var p: Array = entry["pivot_body"]
				pivot = Vector3(p[0], p[1], p[2])
			else:
				var box := _node_aabb(n)
				pivot = Vector3(clampf(0.0, box.position.x, box.end.x), clampf(0.0, box.position.y, box.end.y),
					clampf(0.0, box.position.z, box.end.z))
			hinge_nodes.append([n, Vector3(ax[0], ax[1], ax[2]).normalized(), pivot, n.transform, "joint_" + node_name])


static func _node_aabb(node: Node3D) -> AABB:
	var acc := AABB()
	var first := true
	for mi in [node] + node.find_children("*", "MeshInstance3D", true, false):
		if mi is MeshInstance3D:
			var box := node.transform * _relative_transform(mi as Node3D, node) * (mi as MeshInstance3D).get_aabb()
			acc = box if first else acc.merge(box)
			first = false
	return acc


## Where each hinged part is now (BODY metres), for checks.
func hinge_report() -> Array:
	var out := []
	for h: Array in hinge_nodes:
		if lods[lod].is_ancestor_of(h[0] as Node3D):  # the current level of detail
			var c := _node_aabb(h[0] as Node3D).get_center()
			out.append({"channel": h[4], "center_body": [c.x, c.y, c.z]})
	return out


func set_joint_store(store: RefCounted) -> void:
	_joint_store = store


## Pick a level of detail from the projected radius in pixels.
func choose_lod(radius_px: float) -> void:
	var want := 0 if radius_px > 200.0 else (1 if radius_px > 40.0 else 2)
	lod = mini(want, lods.size() - 1)
	for i in lods.size():
		lods[i].visible = i == lod


## Spin wheel nodes by integrating the logged wheel speed (display only) and
## set hinge angles from joint_<node> channels.
func update_articulation(frame: Dictionary) -> void:
	if _joint_store != null:
		for h: Array in hinge_nodes:
			if _joint_store.has(h[4]):
				var angle: float = _joint_store.value(h[4], frame["t"])[0]
				var pivot: Vector3 = h[2]
				(h[0] as Node3D).transform = Transform3D(Basis(), pivot) * Transform3D(Basis(h[1] as Vector3, angle), Vector3.ZERO) \
					* Transform3D(Basis(), -pivot) * (h[3] as Transform3D)
	if wheel_nodes.is_empty() or not frame.has("wheel_speed"):
		return
	var ts: float = frame["t"]
	var dt := 0.0 if is_nan(_last_t) else clampf(ts - _last_t, -1.0, 1.0)
	_last_t = ts
	var speeds: PackedFloat64Array = frame["wheel_speed"]
	for key: Array in wheel_nodes:
		var i: int = key[0]
		if i >= speeds.size():
			continue
		# Cap the displayed rate below ~1/3 turn per frame so spin never aliases backwards.
		var w := clampf(speeds[i], -20.0, 20.0)
		_wheel_angle[key] = fmod(float(_wheel_angle.get(key, 0.0)) + w * dt, TAU)
		# Spin about the wheel's own centre: the model stores parts at identity
		# transforms with vertices in body coordinates.
		var entry: Array = wheel_nodes[key]
		var pivot: Vector3 = entry[3]
		var spin := Transform3D(Basis(entry[1] as Vector3, _wheel_angle[key]), Vector3.ZERO)
		(entry[0] as Node3D).transform = Transform3D(Basis(), pivot) * spin * Transform3D(Basis(), -pivot) \
			* (entry[2] as Transform3D)
