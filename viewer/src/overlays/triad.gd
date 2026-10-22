extends "res://src/overlays/overlay.gd"
## Body axes (X red, Y green, Z blue), parented to the spacecraft so they
## follow its attitude and scaled-space placement.

func configure(descriptor: Dictionary, scene: Dictionary) -> void:
	super.configure(descriptor, scene)
	var length: float = descriptor.get("length_m", 1.0)
	var mesh := ImmediateMesh.new()
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.vertex_color_use_as_albedo = true
	mat.no_depth_test = true
	mesh.surface_begin(Mesh.PRIMITIVE_LINES, mat)
	for axis: Array in [[Vector3.RIGHT, Color.RED], [Vector3.UP, Color.GREEN], [Vector3.BACK, Color(0.3, 0.5, 1.0)]]:
		mesh.surface_set_color(axis[1])
		mesh.surface_add_vertex(Vector3.ZERO)
		mesh.surface_set_color(axis[1])
		mesh.surface_add_vertex((axis[0] as Vector3) * length)
	mesh.surface_end()
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(mi)
