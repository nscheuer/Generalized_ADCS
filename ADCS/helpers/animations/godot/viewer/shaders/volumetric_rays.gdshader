// Subtle screen-space shafts from the Sun (GPU Gems 3, Chapter 13).
shader_type canvas_item;

uniform vec2 sun_uv = vec2(-1.0);
uniform float rays_visibility = 0.0;
uniform float rays_strength = 0.0;
uniform float rays_density = 0.72;
uniform float rays_decay = 0.96;
uniform float rays_weight = 0.08;
uniform float brightness_cutoff = 0.8;

void fragment() {
	vec4 source = COLOR;
	float edge_distance = min(min(sun_uv.x, 1.0 - sun_uv.x), min(sun_uv.y, 1.0 - sun_uv.y));
	float edge_fade = smoothstep(0.0, 0.08, edge_distance);
	float strength = rays_strength * rays_visibility * edge_fade;
	if (strength > 0.0001) {
		vec2 step_uv = (sun_uv - UV) * (rays_density / 64.0);
		vec2 sample_uv = UV;
		float illumination = 1.0;
		vec3 shafts = vec3(0.0);
		for (int i = 0; i < 64; i++) {
			sample_uv += step_uv;
			vec3 sample_color = texture(TEXTURE, sample_uv).rgb;
			float luminance = dot(sample_color, vec3(0.2126, 0.7152, 0.0722));
			float contribution = max(luminance - brightness_cutoff, 0.0) / max(luminance, 0.001);
			shafts += sample_color * contribution * illumination * rays_weight;
			illumination *= rays_decay;
		}
		source.rgb += shafts * strength;
	}
	COLOR = source;
}
