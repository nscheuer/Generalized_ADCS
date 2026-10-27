shader_type spatial;
render_mode unshaded, cull_disabled, blend_mix, depth_prepass_alpha;

uniform vec3 sun_dir_world = vec3(1.0, 0.0, 0.0);

void fragment() {
	float r = length(UV * 2.0 - 1.0);
	if (r < 0.49 || r > 1.0) { discard; }
	float bands = 0.58 + 0.28 * sin(r * 150.0) + 0.10 * sin(r * 533.0);
	float edge = smoothstep(0.49, 0.52, r) * (1.0 - smoothstep(0.96, 1.0, r));
	ALBEDO = vec3(0.78, 0.68, 0.52) * max(0.12, abs(sun_dir_world.z));
	ALPHA = clamp(bands * edge, 0.0, 0.85);
}
