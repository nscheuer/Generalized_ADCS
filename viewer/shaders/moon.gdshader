// Moon: colour and object-space normal cube maps, Lambert lighting in object space.
shader_type spatial;
render_mode unshaded, cull_back;

uniform samplerCube color_tex : source_color, filter_linear_mipmap;
uniform samplerCube normal_tex : filter_linear_mipmap;
uniform bool has_color = false;
uniform bool has_normals = false;
uniform vec3 albedo = vec3(0.12);
uniform vec3 sun_dir_local = vec3(1.0, 0.0, 0.0);
uniform float ambient = 0.004;

varying vec3 p_local;

void vertex() {
	p_local = VERTEX;
}

void fragment() {
	vec3 d = normalize(p_local);
	vec3 n = has_normals ? normalize(texture(normal_tex, d).xyz * 2.0 - 1.0) : d;
	vec3 base = has_color ? texture(color_tex, d).rgb : albedo;
	ALBEDO = base * (max(dot(n, sun_dir_local), 0.0) + ambient);
}
