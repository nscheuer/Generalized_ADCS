// Screen-space shafts based on the GPU Gems 3 radial-scattering post-process.
// The rendered Sun provides the bright source; the Earth and spacecraft occlude
// it in the source image and therefore interrupt the accumulated shafts.
shader_type canvas_item;

uniform vec2 sun_uv = vec2(-1.0);
uniform bool rays_enabled = false;
uniform float rays_strength = 1.1;
uniform float rays_density = 0.7;
uniform float rays_decay = 0.96;
uniform float rays_weight = 0.3;
uniform float brightness_cutoff = 0.65;

void fragment() {
	vec4 source = COLOR;
	if (rays_enabled) {
		vec2 step_uv = (sun_uv - UV) * (rays_density / 128.0);
		vec2 sample_uv = UV;
		float illumination = 1.0;
		vec3 shafts = vec3(0.0);
		for (int i = 0; i < 128; i++) {
			sample_uv += step_uv;
			vec3 sample_color = texture(TEXTURE, sample_uv).rgb;
			float brightness = max(dot(sample_color, vec3(0.2126, 0.7152, 0.0722)) - brightness_cutoff, 0.0);
			shafts += sample_color * brightness * illumination * rays_weight;
			illumination *= rays_decay;
		}
		source.rgb += shafts * rays_strength;
	}
	COLOR = source;
}
