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
	float overlap_m = max(3000.0, 1.5 * fwidth(closest_r));
	vec2 top = atm_ray_sphere(cam_local, d, atm_top);
	if (!has_atmosphere || top.y < 0.0) {
		discard;
	}
	vec2 ground = atm_ray_sphere(cam_local, d, atm_bottom);
	float end_t = top.y;
	float limb_coverage = 1.0;
	if (ground.x > 0.0) {
		// Fade the shell where it meets the rasterized Earth silhouette. A hard
		// discard at this derivative-based boundary can pop as the camera pans,
		// especially when the horizon spans many pixels on AMD/WSL renderers.
		limb_coverage = smoothstep(atm_bottom - overlap_m, atm_bottom + overlap_m, closest_r);
	}
	if (ground.x > 0.0) {
		// The Earth mesh covers most ground-facing rays. Only fill the thin
		// silhouette mismatch; never march through the solid planet.
		end_t = min(end_t, ground.x);
	}
	vec3 L;
	vec3 T;
	atm_integrate(cam_local, d, max(top.x, 0.0), end_t, sun_dir_itrs, L, T);
	ALBEDO = L * PI * sun_scale * limb_coverage;
	ALPHA = clamp(1.0 - dot(T, vec3(1.0 / 3.0)), 0.0, 1.0) * limb_coverage;  // stars behind the limb dim
}
