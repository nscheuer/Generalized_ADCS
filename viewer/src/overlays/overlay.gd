extends Node3D
## Base class for overlays (vectors, cones, tracks, labels) drawn on top of the
## scene. Phase 7 adds the full set; descriptors will come from Python
## (ADCS.viz.overlays). Subclasses implement configure() and update().

var desc: Dictionary = {}


## Called once with the overlay descriptor and the vizlog scene description.
func configure(descriptor: Dictionary, _scene: Dictionary) -> void:
	desc = descriptor


## Called every frame after the world has been posed.
func update(_frame: Dictionary, _world: RefCounted) -> void:
	pass
