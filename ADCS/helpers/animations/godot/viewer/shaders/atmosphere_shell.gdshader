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
	// Direct sunlight blends the lower limb toward the visible daylight horizon.
	// As the Sun approaches the local horizon this correction fades out,
	// preserving the warm twilight scattering.
	float direct_sunlight = smoothstep(0.0, 0.30, sun_elevation);
	float limb_coverage = smoothstep(atm_bottom - overlap_m, atm_bottom + overlap_m, closest_r);
	// Brighten and saturate only the spaceward gradient. The increase with
	// altitude keeps the warm lower twilight and daylight surface match intact.
	float upper_gradient = smoothstep(atm_bottom + 12000.0, atm_top, closest_r);
	vec3 physical_shell_color = min(L * PI * sun_scale * (1.0 + 1.6 * upper_gradient)
		* vec3(0.88, 1.0, 1.34), vec3(1.0));
	// Preserve coverage and opacity. Premultiplied alpha means the color
	// contribution must compensate for the limb's antialias coverage; otherwise
	// the half-covered tangent pixels form a dark line against the surface.
	float lower_atmosphere = 1.0 - smoothstep(atm_bottom + 5000.0, min(atm_bottom + 70000.0, atm_top), closest_r);
	float atmosphere_alpha = clamp(1.0 - dot(T, vec3(1.0 / 3.0)), 0.0, 1.0);
	float daylight_merge = direct_sunlight * lower_atmosphere;
	vec3 horizon_match_color = vec3(0.45, 0.40, 0.48) / max(limb_coverage, 0.20);
	ALBEDO = mix(physical_shell_color, horizon_match_color, daylight_merge) * limb_coverage;
	ALPHA = atmosphere_alpha * limb_coverage;  // stars behind the limb dim
}
