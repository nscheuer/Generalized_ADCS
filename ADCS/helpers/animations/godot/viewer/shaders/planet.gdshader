shader_type spatial;
render_mode unshaded, cull_back;

uniform vec4 base_color : source_color = vec4(1.0);
uniform vec3 sun_dir_world = vec3(1.0, 0.0, 0.0);
uniform bool has_texture = false;
uniform sampler2D albedo_texture : source_color, filter_linear_mipmap;

void fragment() {
	vec3 world_normal = normalize((INV_VIEW_MATRIX * vec4(NORMAL, 0.0)).xyz);
	float lit = max(dot(world_normal, normalize(sun_dir_world)), 0.0);
	vec3 surface = has_texture ? texture(albedo_texture, UV).rgb : base_color.rgb;
	ALBEDO = surface * (0.015 + 0.985 * lit);
}
