extends RefCounted
## Overlay type name -> script. Phase 7 adds arrow, cone, polyline, target_line, label.

const TYPES := {
	"triad": preload("res://src/overlays/triad.gd"),
}


static func create(descriptor: Dictionary, scene: Dictionary) -> Node3D:
	var script: GDScript = TYPES.get(descriptor.get("type", ""))
	if script == null:
		push_warning("unknown overlay type %s" % descriptor.get("type"))
		return null
	var node: Node3D = script.new()
	node.configure(descriptor, scene)
	return node
