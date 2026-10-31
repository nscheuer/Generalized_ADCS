shader_type spatial;
render_mode unshaded, cull_back;

uniform vec4 base_color : source_color = vec4(1.0);
uniform vec3 sun_dir_world = vec3(1.0, 0.0, 0.0);
uniform bool has_texture = false;
uniform sampler2D albedo_texture : source_color, filter_linear_mipmap;
uniform vec3 haze_color : source_color = vec3(0.8, 0.85, 1.0);
uniform float haze_strength = 0.0;
uniform bool gas_giant = false;
uniform vec3 band_color : source_color = vec3(0.6, 0.5, 0.4);
uniform float band_strength = 0.0;

varying vec3 planet_normal_world;

void vertex() {
	planet_normal_world = normalize(MODEL_NORMAL_MATRIX * NORMAL);
}

void fragment() {
	float lit = smoothstep(-0.08, 0.32, dot(normalize(planet_normal_world), normalize(sun_dir_world)));
	vec3 surface = has_texture ? texture(albedo_texture, UV).rgb : base_color.rgb;
	if (gas_giant && !has_texture) {
		float latitude = UV.y * PI;
		float warp = 0.006 * sin(UV.x * 2.0 * PI * 5.0 + latitude * 3.0);
		float stripes = sin((UV.y + warp) * 65.0) + 0.33 * sin((UV.y - warp) * 143.0);
		float band = smoothstep(-0.25, 0.7, stripes) * band_strength;
		surface = mix(surface, band_color, band);
	}
	float rim = pow(1.0 - max(dot(normalize(NORMAL), normalize(VIEW)), 0.0), 2.0);
	surface = mix(surface, haze_color, haze_strength * rim);
	ALBEDO = surface * (0.025 + 0.975 * lit);
}
