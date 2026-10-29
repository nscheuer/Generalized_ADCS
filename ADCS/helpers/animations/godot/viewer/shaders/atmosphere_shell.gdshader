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
	// The upper veil's daylight treatment fades with altitude and direct Sun
	// incidence, leaving the dusk/sunrise profile untouched.
	float upper_gradient = smoothstep(atm_bottom + 5000.0, atm_top, closest_r);
	float bright_sunlight = smoothstep(0.35, 0.80, sun_elevation);
	vec3 physical_shell_color = min(L * PI * sun_scale, vec3(1.0));
	vec3 sunlit_blue_color = physical_shell_color * vec3(5, 140, 213);
	physical_shell_color = mix(physical_shell_color, sunlit_blue_color, bright_sunlight * upper_gradient);
	// Premultiplied alpha means the color contribution must compensate for the
	// limb's antialias coverage; otherwise the half-covered tangent pixels form
	// a dark line against the surface.
	float lower_atmosphere = 1.0 - smoothstep(atm_bottom + 5000.0, min(atm_bottom + 70000.0, atm_top), closest_r);
	float atmosphere_alpha = clamp(1.0 - dot(T, vec3(1.0 / 3.0)), 0.0, 1.0);
	float daylight_merge = direct_sunlight * lower_atmosphere;
	vec3 horizon_match_color = vec3(0.72, 0.70, 0.92) / max(limb_coverage, 0.20);
	ALBEDO = mix(physical_shell_color, horizon_match_color, daylight_merge) * limb_coverage;
	// Keep the physical dusk opacity curve. In strong direct sunlight, gently
	// raise the upper veil toward a visible but still translucent 35% opacity.
	atmosphere_alpha = mix(atmosphere_alpha, max(atmosphere_alpha, 0.01 * upper_gradient), bright_sunlight);
	ALPHA = atmosphere_alpha * limb_coverage;  // stars behind the limb dim
}
