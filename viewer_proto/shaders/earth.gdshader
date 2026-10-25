// Prototype Earth: Lambert day side, optional texture, dark night side.
// Mesh vertices are in ITRS (metres); lat/lon come from the local position,
// so the texture is aligned by Earth orientation alone.
shader_type spatial;
render_mode unshaded, cull_back;

uniform sampler2D day_tex : source_color, filter_linear, repeat_enable;
uniform bool use_texture = false;
uniform vec3 albedo = vec3(1.0);
uniform vec3 sun_dir_itrs = vec3(1.0, 0.0, 0.0);
uniform float ambient = 0.0;

varying vec3 p_itrs;

void vertex() {
	p_itrs = VERTEX;
}

void fragment() {
	vec3 n = normalize(p_itrs);
	float lon = atan(n.y, n.x);
	float lat = asin(clamp(n.z, -1.0, 1.0));
	vec2 uv = vec2(lon / (2.0 * PI) + 0.5, 0.5 - lat / PI);
	vec3 base = use_texture ? texture(day_tex, uv).rgb : albedo;
	ALBEDO = base * (max(dot(n, sun_dir_itrs), 0.0) + ambient);
}
