extends RefCounted
## Overlay type name (scene.json "type") -> script.

const TYPES := {
	"arrow": preload("res://src/overlays/arrow.gd"),
	"triad": preload("res://src/overlays/triad.gd"),
	"cone": preload("res://src/overlays/cone.gd"),
	"polyline": preload("res://src/overlays/track.gd"),
	"target_line": preload("res://src/overlays/target_line.gd"),
	"label": preload("res://src/overlays/label.gd"),
}
## Drawn at the spacecraft (children of its node); the rest live in the scene root.
const ATTACHED := ["arrow", "triad", "cone", "label"]


static func create(descriptor: Dictionary, scene: Dictionary) -> Node3D:
	var script: GDScript = TYPES.get(descriptor.get("type", ""))
	if script == null:
		push_warning("unknown overlay type %s" % descriptor.get("type"))
		return null
	var node: Node3D = script.new()
	node.configure(descriptor, scene)
	return node
