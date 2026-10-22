// Unshaded constant colour; values above 1 test HDR headroom.
shader_type spatial;
render_mode unshaded, cull_disabled;

uniform vec3 color = vec3(1.0);
uniform bool as_emission = false;

void fragment() {
	if (as_emission) {
		ALBEDO = vec3(0.0);
		EMISSION = color;
	} else {
		ALBEDO = color;
	}
}
