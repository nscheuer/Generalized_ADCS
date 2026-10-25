// The atmosphere seen against space (limb glow): rays that pass through the
// shell without reaching the ground. Rays that hit the ground are handled by
// earth.gdshader (aerial perspective), so they are discarded here.
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
	vec2 top = atm_ray_sphere(cam_local, d, atm_top);
	vec2 ground = atm_ray_sphere(cam_local, d, atm_bottom);
	if (!has_atmosphere || top.y < 0.0 || ground.x > 0.0) {
		discard;
	}
	vec3 L;
	vec3 T;
	atm_integrate(cam_local, d, max(top.x, 0.0), top.y, sun_dir_itrs, L, T);
	ALBEDO = L * PI * sun_scale;
	ALPHA = clamp(1.0 - dot(T, vec3(1.0 / 3.0)), 0.0, 1.0);  // stars behind the limb dim
}
