// Moon: colour and object-space normal cube maps. Lommel-Seeliger reflectance
// (the classic airless-body law: no limb darkening at full Moon), normalised so
// it equals Lambert at normal incidence and emission.
shader_type spatial;
render_mode unshaded, cull_back;

uniform samplerCube color_tex : source_color, filter_linear_mipmap;
uniform samplerCube normal_tex : filter_linear_mipmap;
uniform bool has_color = false;
uniform bool has_normals = false;
uniform vec3 albedo = vec3(0.12);
uniform vec3 sun_dir_local = vec3(1.0, 0.0, 0.0);
uniform vec3 cam_local = vec3(-4.0e8, 0.0, 0.0);
uniform float ambient = 0.004;   // earthshine on the night side

varying vec3 p_local;

void vertex() {
	p_local = VERTEX;
}

void fragment() {
	vec3 d = normalize(p_local);
	vec3 n = has_normals ? normalize(texture(normal_tex, d).xyz * 2.0 - 1.0) : d;
	vec3 base = has_color ? texture(color_tex, d).rgb : albedo;
	float mu0 = max(dot(n, sun_dir_local), 0.0);
	float mu = max(dot(n, normalize(cam_local - p_local)), 1e-3);
	ALBEDO = base * (2.0 * mu0 / (mu0 + mu) + ambient);
}
