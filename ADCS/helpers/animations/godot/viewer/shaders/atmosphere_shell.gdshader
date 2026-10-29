// The atmosphere seen against space (limb glow): rays that pass through the
// shell without reaching the ground. A narrow overlap at the limb covers the
// gap between the analytic ground sphere and the rasterized Earth mesh.
shader_type spatial;
render_mode unshaded, cull_front, depth_draw_never, blend_premul_alpha, shadows_disabled;

#include "res://shaders/atmosphere.gdshaderinc"

uniform vec3 sun_dir_itrs = vec3(1.0, 0.0, 0.0);
uniform float sun_scale = 1.0;

varying vec3 p_local;

void vertex() {
	p_local = VERTEX;
}

void fragment() {
	vec3 d = normalize(p_local - cam_local);
	float closest_r = length(cam_local - d * dot(cam_local, d));
	float overlap_m = max(30000.0, 3.0 * fwidth(closest_r));
	vec2 top = atm_ray_sphere(cam_local, d, atm_top);
	if (!has_atmosphere || top.y < 0.0) {
		discard;
	}
	vec2 ground = atm_ray_sphere(cam_local, d, atm_bottom);
	float end_t = top.y;
	if (ground.x > 0.0) {
		end_t = min(end_t, ground.x);
	}
	vec3 L;
	vec3 T;
	atm_integrate(cam_local, d, max(top.x, 0.0), end_t, sun_dir_itrs, L, T);
	vec3 limb_point = cam_local - d * dot(cam_local, d);
	float sun_elevation = dot(normalize(limb_point), sun_dir_itrs);
	float daylight = smoothstep(0.05, 0.40, sun_elevation);
	float limb_coverage = smoothstep(atm_bottom - overlap_m, atm_bottom + overlap_m, closest_r);
	// Preserve coverage and opacity. Drive the daylight limb hue from its
	// actual scattered radiance: the brighter the lower limb, the warmer it
	// becomes. The color shift fades smoothly into the natural blue above it.
	float lower_atmosphere = 1.0 - smoothstep(atm_bottom + 5000.0, min(atm_bottom + 70000.0, atm_top), closest_r);
	vec3 daylight_color = min(L * PI * sun_scale, vec3(1.0));
	float limb_brightness = dot(daylight_color, vec3(0.2126, 0.7152, 0.0722));
	float brightness_warmth = smoothstep(0.04, 0.45, limb_brightness);
	float warm_tint = daylight * lower_atmosphere * brightness_warmth * 0.75;
	daylight_color *= mix(vec3(1.0), vec3(1.30, 0.90, 0.68), warm_tint);
	ALBEDO = daylight_color * limb_coverage;
	ALPHA = clamp(1.0 - dot(T, vec3(1.0 / 3.0)), 0.0, 1.0) * limb_coverage;  // stars behind the limb dim
}
