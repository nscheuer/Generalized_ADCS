// UV.x is distance from Saturn's centre in thousands of kilometres.
shader_type spatial;
render_mode unshaded, cull_disabled, blend_mix, depth_draw_never;

uniform vec3 sun_dir_local = vec3(1.0, 0.0, 0.0);
uniform float saturn_radius = 60268000.0;

varying vec3 ring_point;

void vertex() {
	ring_point = VERTEX;
}

void fragment() {
	float r = UV.x;
	vec3 color = vec3(0.0);
	float opacity = 0.0;
	if (r < 92.0) { // translucent C ring
		color = vec3(0.65, 0.59, 0.50);
		opacity = 0.28;
	} else if (r < 117.58) { // bright B ring
		color = vec3(0.89, 0.83, 0.70);
		opacity = 0.88;
	} else if (r < 122.17) { // Cassini Division
		color = vec3(0.40, 0.35, 0.30);
		opacity = 0.045;
	} else if (r < 136.78) { // A ring
		color = vec3(0.78, 0.73, 0.64);
		opacity = 0.72;
	} else if (r < 140.18) { // outer A ring and Encke gap
		color = vec3(0.72, 0.68, 0.61);
		opacity = 0.62;
	} else { // F ring
		color = vec3(0.76, 0.72, 0.64);
		opacity = 0.35;
	}
	float encke_gap = 1.0 - 0.93 * smoothstep(133.45, 133.55, r) * (1.0 - smoothstep(133.8, 133.9, r));
	float fine_bands = 0.92 + 0.05 * sin(r * 17.0) + 0.03 * sin(r * 71.0);
	float edge = smoothstep(74.51, 74.8, r) * (1.0 - smoothstep(140.18, 140.23, r));
	vec3 sun = normalize(sun_dir_local);
	float toward_planet = -dot(ring_point, sun);
	float closest = length(ring_point + max(toward_planet, 0.0) * sun);
	float shadow = toward_planet > 0.0 ? smoothstep(saturn_radius * 0.995, saturn_radius * 1.015, closest) : 1.0;
	float light = 0.18 + 0.82 * abs(sun.z) * shadow;
	ALBEDO = color * light * fine_bands;
	ALPHA = clamp(opacity * encke_gap * edge, 0.0, 0.95);
}
