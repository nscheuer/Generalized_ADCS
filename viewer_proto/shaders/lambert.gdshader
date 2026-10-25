// Lambert-lit surface with sunlight scaled by the simulation's shadow factor.
shader_type spatial;
render_mode unshaded, cull_back;

uniform vec3 albedo = vec3(0.6);
uniform vec3 sun_dir_world = vec3(1.0, 0.0, 0.0);
uniform float sun_scale = 1.0;
uniform float ambient = 0.03;

void fragment() {
	vec3 n_world = normalize((INV_VIEW_MATRIX * vec4(NORMAL, 0.0)).xyz);
	ALBEDO = albedo * (max(dot(n_world, sun_dir_world), 0.0) * sun_scale + ambient);
}
