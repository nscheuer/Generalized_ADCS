extends "res://src/overlays/overlay.gd"
## Text pinned to the spacecraft (always readable, constant size on screen).

var label: Label3D


func configure(descriptor: Dictionary, scene_desc: Dictionary) -> void:
	super.configure(descriptor, scene_desc)
	label = make_label(String(desc.get("text", "")), color_of(desc.get("color"), Color.WHITE))
	add_child(label)


func update(ctx: Dictionary) -> void:
	var px_per_m: float = ctx["focal"] / maxf(ctx["d_craft"], 1e-9)
	var off: Array = desc.get("offset_px", [0.0, -40.0])
	# Offset along the camera's up direction, expressed in BODY coordinates.
	var rot: PackedFloat64Array = ctx["rot"]
	var cam_rot: PackedFloat64Array = ctx["cam_rot"]
	var up := D.mat_t_vec(rot, PackedFloat64Array([cam_rot[1], cam_rot[4], cam_rot[7]]))
	var right := D.mat_t_vec(rot, PackedFloat64Array([cam_rot[0], cam_rot[3], cam_rot[6]]))
	var o := D.add(D.scale(right, float(off[0]) / px_per_m), D.scale(up, -float(off[1]) / px_per_m))
	label.position = Vector3(o[0], o[1], o[2])
	fit_label(label, ctx["focal"], ctx["k_craft"])
