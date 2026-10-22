// Sun disk with limb darkening and a soft halo; HDR values feed the glow pass.
shader_type spatial;
render_mode unshaded, cull_disabled, blend_add, depth_draw_never, depth_test_disabled;

uniform float intensity = 40.0;
uniform float disk_fraction = 0.35;  // disk radius / quad half-size

void fragment() {
	float r = length(UV * 2.0 - 1.0) / disk_fraction;
	float mu = sqrt(max(1.0 - r * r, 0.0));
	float disk = r < 1.0 ? (1.0 - 0.6 * (1.0 - mu)) : 0.0;  // linear limb darkening
	float halo = exp(-4.0 * max(r - 1.0, 0.0)) * 0.08;
	ALBEDO = vec3(1.0, 0.97, 0.92) * intensity * (disk + halo);
}
