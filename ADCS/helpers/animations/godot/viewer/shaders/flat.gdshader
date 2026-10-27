// Unshaded constant colour (debug geometry, overlays).
shader_type spatial;
render_mode unshaded, cull_disabled;

uniform vec3 color = vec3(1.0);

void fragment() {
	ALBEDO = color;
}
