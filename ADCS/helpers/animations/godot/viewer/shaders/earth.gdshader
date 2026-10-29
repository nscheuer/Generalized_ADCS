// Earth surface. Mesh vertices and light directions are in ITRS metres. The
// Solar System Scope maps use equirectangular longitude/latitude coordinates;
// locally built products use object-space cube maps. Both stay fixed to Earth
// as the vizlog's ITRS orientation changes. Lighting is computed in object
// space because the body is drawn in scaled space.
//
// Display units: a white Lambertian surface under the full Sun reads 1.
shader_type spatial;
// Render both sides of the sphere. On some WSL GPU drivers cull_back removes
// the visible Earth surface while the separately rendered atmosphere remains.
// Opaque depth testing still hides the far hemisphere.
render_mode unshaded, cull_disabled;

#include "res://shaders/atmosphere.gdshaderinc"

uniform samplerCube day_tex : source_color, filter_linear_mipmap;
uniform samplerCube night_tex : source_color, filter_linear_mipmap;
uniform samplerCube cloud_tex : filter_linear_mipmap;
uniform samplerCube normal_tex : filter_linear_mipmap;
uniform samplerCube ocean_tex : filter_linear_mipmap;
uniform sampler2D day_map : source_color, filter_linear_mipmap, repeat_enable;
uniform sampler2D night_map : source_color, filter_linear_mipmap, repeat_enable;
uniform sampler2D cloud_map : filter_linear_mipmap, repeat_enable;
uniform sampler2D normal_map : filter_linear_mipmap, repeat_enable;
uniform sampler2D specular_map : filter_linear_mipmap, repeat_enable;
uniform bool has_day = false;
uniform bool has_night = false;
uniform bool has_clouds = false;
uniform bool has_normals = false;
uniform bool has_ocean = false;
uniform bool has_day_map = false;
uniform bool has_night_map = false;
uniform bool has_cloud_map = false;
uniform bool has_normal_map = false;
uniform bool has_specular_map = false;
uniform vec3 albedo = vec3(0.3, 0.4, 0.6);      // used without a day map (and by checks)
uniform vec3 sun_dir_itrs = vec3(1.0, 0.0, 0.0);
uniform float sun_scale = 1.0;
uniform float night_gain = 0.45;
uniform vec3 night_tint = vec3(1.0, 0.78, 0.52);  // sodium/LED mix
uniform float cloud_opacity = 0.9;
uniform float cloud_height = 6000.0;     // m, for cloud shadows
uniform float cloud_shadow = 0.55;
uniform float ambient = 0.0;
uniform float specular_strength = 0.6;

// Aerial perspective is computed per vertex (it varies smoothly over the
// disk; ~27x fewer evaluations than per pixel) and per pixel only where the
// view grazes the surface near the limb, where it changes steeply.
uniform float limb_band_lo = 0.18;   // cos(view angle to the surface normal)
uniform float limb_band_hi = 0.32;
uniform int pixel_steps = 16;

varying vec3 p_itrs;
varying vec3 ap_L;
varying vec3 ap_T;

vec2 equirect_uv(vec3 n) {
	// ITRS +X is Greenwich (u = 0.5), +Y is 90° east, +Z is north.
	return vec2(atan(n.y, n.x) / (2.0 * PI) + 0.5, 0.5 - asin(clamp(n.z, -1.0, 1.0)) / PI);
}

vec3 sample_day_map(vec2 uv, vec2 uv_dx, vec2 uv_dy) {
	vec3 color = textureGrad(day_map, uv, uv_dx, uv_dy).rgb;
	float edge = min(uv.x, 1.0 - uv.x);
	// The equirectangular texture's first and last columns meet over the
	// Pacific. Blend their nearby interior texels across this narrow seam to
	// suppress bright border pixels and mipmap/filtering artifacts.
	if (edge < 0.003) {
		float seam = 1.0 - smoothstep(0.0, 0.003, edge);
		vec2 edge_dx = vec2(0.0, uv_dx.y);
		vec2 edge_dy = vec2(0.0, uv_dy.y);
		vec3 left_edge = textureGrad(day_map, vec2(0.003, uv.y), edge_dx, edge_dy).rgb;
		vec3 right_edge = textureGrad(day_map, vec2(0.997, uv.y), edge_dx, edge_dy).rgb;
		color = mix(color, 0.5 * (left_edge + right_edge), seam);
	}
	return color;
}

float cloud_coverage(vec3 n) {
	if (has_cloud_map) {
		return texture(cloud_map, equirect_uv(n)).r;
	}
	return has_clouds ? texture(cloud_tex, n).r : 0.0;
}

void vertex() {
	p_itrs = VERTEX;
	ap_L = vec3(0.0);
	ap_T = vec3(1.0);
	if (has_atmosphere) {
		atm_aerial(VERTEX, sun_dir_itrs, atm_steps, ap_L, ap_T);
	}
}

void fragment() {
	vec3 n_geo = normalize(p_itrs);
	vec2 uv = equirect_uv(n_geo);
	vec3 n = n_geo;
	vec2 uv_dx = dFdx(uv);
	vec2 uv_dy = dFdy(uv);
	// atan longitude wraps from 1 to 0. Remove that integer jump from the
	// horizontal derivatives so mip selection stays continuous across the seam.
	uv_dx.x -= round(uv_dx.x);
	uv_dy.x -= round(uv_dy.x);
	if (has_normal_map) {
		vec3 tangent = vec3(0.0, 1.0, 0.0);
		if (abs(n_geo.z) < 0.999) {
			tangent = normalize(vec3(-n_geo.y, n_geo.x, 0.0));
		}
		vec3 bitangent = normalize(cross(n_geo, tangent));
		vec3 tangent_normal = texture(normal_map, uv).xyz * 2.0 - 1.0;
		n = normalize(tangent * tangent_normal.x + bitangent * tangent_normal.y + n_geo * tangent_normal.z);
	} else if (has_normals) {
		n = normalize(texture(normal_tex, n_geo).xyz * 2.0 - 1.0);
	}
	vec3 base = has_day_map ? sample_day_map(uv, uv_dx, uv_dy) : (has_day ? texture(day_tex, n_geo).rgb : albedo);
	float mu_geo = dot(n_geo, sun_dir_itrs);

	// Sunlight reaching the ground: attenuated (and reddened) by the atmosphere.
	vec3 sun_ground = vec3(1.0);
	if (has_atmosphere) {
		sun_ground = atm_transmittance(atm_bottom, mu_geo) * atm_sun_visible(atm_bottom + 1.0, mu_geo);
	}
	// Let the atmosphere carry sunlight into civil twilight instead of ending
	// it at a nearly geometric one-degree terminator.
	float terminator = smoothstep(-0.08, 0.04, mu_geo);
	float cloud_light = 1.0;

	float cloud = cloud_coverage(n_geo);
	if ((has_clouds || has_cloud_map) && mu_geo > 0.0) {
		// Shadow of the cloud layer: look up the cloud where the ray toward the Sun crosses it.
		vec3 q = normalize(n_geo * atm_bottom + sun_dir_itrs * (cloud_height / max(mu_geo, 0.05)));
		cloud_light = 1.0 - cloud_shadow * cloud_coverage(q) * cloud_opacity;
	}
	vec3 sun_irradiance = terminator * sun_scale * sun_ground * cloud_light;
	vec3 lit = max(dot(n, sun_dir_itrs), 0.0) * sun_irradiance;
	vec3 color = base * (lit + ambient);

	vec3 view = normalize(cam_local - p_itrs);
	if (has_ocean || has_specular_map) {
		// Sun glint on water (Blinn-Phong).
		vec3 h = normalize(view + sun_dir_itrs);
		float water = has_specular_map ? texture(specular_map, uv).r : texture(ocean_tex, n_geo).r;
		color += water * specular_strength * pow(max(dot(n_geo, h), 0.0), 120.0) * sun_irradiance;
	}
	if (has_clouds || has_cloud_map) {
		vec3 cloud_lit = max(mu_geo, 0.0) * terminator * sun_scale * sun_ground + ambient;
		color = mix(color, cloud_lit, cloud * cloud_opacity);
	}
	if (has_night || has_night_map) {
		// City lights fade in across twilight (sun 3 deg above to 9 deg below the horizon).
		float dark = 1.0 - smoothstep(-0.15, 0.05, mu_geo);
		vec3 night = has_night_map ? texture(night_map, uv).rgb : texture(night_tex, n_geo).rgb;
		color += night * night_tint * night_gain * dark * (1.0 - 0.7 * cloud);
	}

	if (has_atmosphere) {
		// Aerial perspective: the atmosphere between the camera and this point.
		vec3 L = ap_L;
		vec3 T = ap_T;
		float grazing = dot(n_geo, view);
		if (grazing < limb_band_hi) {
			vec3 Lp;
			vec3 Tp;
			atm_aerial(p_itrs, sun_dir_itrs, pixel_steps, Lp, Tp);
			float w = smoothstep(limb_band_lo, limb_band_hi, grazing);
			L = mix(Lp, L, w);
			T = mix(Tp, T, w);
		}
		color = color * T + L * PI * sun_scale;
	}
	ALBEDO = color;
}
