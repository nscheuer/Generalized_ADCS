// Orbit / ground track line: constant colour, drawn only between t0 and t1
// (per-vertex time in UV.x, seconds relative to the mesh's reference time).
shader_type spatial;
render_mode unshaded, cull_disabled;

uniform vec3 color = vec3(1.0);
uniform float t0 = -1e30;
uniform float t1 = 1e30;

varying float vt;

void vertex() {
	vt = UV.x;
}

void fragment() {
	if (vt < t0 || vt > t1) {
		discard;
	}
	ALBEDO = color;
}
