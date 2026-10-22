// Earth surface. Mesh vertices are in ITRS metres and every map is a cube map
// sampled with the ITRS direction, so the texture follows Earth orientation
// from the vizlog alone. Lighting is done here (unshaded) because the body is
// drawn in scaled space; all vectors are in object (ITRS) space.
// Atmosphere and cloud shadows come in Phase 6.
shader_type spatial;
render_mode unshaded, cull_back;

uniform samplerCube day_tex : source_color, filter_linear_mipmap;
uniform samplerCube night_tex : source_color, filter_linear_mipmap;
uniform samplerCube cloud_tex : filter_linear_mipmap;
uniform samplerCube normal_tex : filter_linear_mipmap;
uniform samplerCube ocean_tex : filter_linear_mipmap;
uniform bool has_day = false;
uniform bool has_night = false;
uniform bool has_clouds = false;
uniform bool has_normals = false;
uniform bool has_ocean = false;
uniform vec3 albedo = vec3(0.3, 0.4, 0.6);      // used without a day map (and by checks)
uniform vec3 sun_dir_itrs = vec3(1.0, 0.0, 0.0);
uniform float sun_scale = 1.0;
uniform float night_gain = 0.45;
uniform vec3 night_tint = vec3(1.0, 0.78, 0.52);  // sodium/LED mix
uniform float cloud_opacity = 0.9;
uniform float ambient = 0.0;
uniform float specular_strength = 0.6;

varying vec3 p_itrs;

void vertex() {
	p_itrs = VERTEX;
}

void fragment() {
	vec3 n_geo = normalize(p_itrs);
	vec3 n = n_geo;
	if (has_normals) {
		n = normalize(texture(normal_tex, n_geo).xyz * 2.0 - 1.0);
	}
	vec3 base = has_day ? texture(day_tex, n_geo).rgb : albedo;
	float mu_geo = dot(n_geo, sun_dir_itrs);
	float lit = max(dot(n, sun_dir_itrs), 0.0) * smoothstep(-0.02, 0.02, mu_geo) * sun_scale;
	vec3 color = base * (lit + ambient);

	if (has_ocean) {
		// Sun glint on water (Blinn-Phong), view vector in object space.
		vec3 cam = (inverse(MODEL_MATRIX) * vec4(CAMERA_POSITION_WORLD, 1.0)).xyz;
		vec3 v = normalize(cam - p_itrs);
		vec3 h = normalize(v + sun_dir_itrs);
		float water = texture(ocean_tex, n_geo).r;
		color += water * specular_strength * pow(max(dot(n_geo, h), 0.0), 120.0) * lit;
	}
	if (has_clouds) {
		float c = texture(cloud_tex, n_geo).r * cloud_opacity;
		color = mix(color, vec3(max(mu_geo, 0.0) * sun_scale + ambient), c);
	}
	if (has_night) {
		// City lights fade in across twilight (sun 3 deg above to 9 deg below the horizon).
		float dark = smoothstep(0.05, -0.15, mu_geo);
		float cloud = has_clouds ? texture(cloud_tex, n_geo).r : 0.0;
		color += texture(night_tex, n_geo).rgb * night_tint * night_gain * dark * (1.0 - 0.7 * cloud);
	}
	ALBEDO = color;
}
