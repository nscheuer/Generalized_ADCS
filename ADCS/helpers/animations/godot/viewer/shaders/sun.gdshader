// Sun disk with limb darkening and a soft halo; HDR values feed the glow pass.
// Depth-tested: drawn just inside the far plane, behind every far body (scaled
// space puts their centres at <= 0.9 x far), so Earth, Moon and the spacecraft hide it.
shader_type spatial;
render_mode unshaded, cull_disabled, blend_add, depth_draw_never;

uniform float intensity = 40.0;
uniform float disk_fraction = 0.35;  // disk radius / quad half-size
uniform sampler2D surface_map : source_color, filter_linear_mipmap, repeat_enable;
uniform bool has_surface_map = false;
uniform mat3 camera_to_sun = mat3(1.0);

void fragment() {
	float r = length(UV * 2.0 - 1.0) / disk_fraction;
	float mu = sqrt(max(1.0 - r * r, 0.0));
	float disk = r < 1.0 ? (1.0 - 0.6 * (1.0 - mu)) : 0.0;  // linear limb darkening
	float halo = exp(-4.0 * max(r - 1.0, 0.0)) * 0.08;
	vec3 surface = vec3(1.0, 0.97, 0.92);
	if (has_surface_map && r < 1.0) {
		vec2 xy = (UV * 2.0 - 1.0) / disk_fraction;
		vec3 normal_body = normalize(camera_to_sun * vec3(xy, mu));
		vec2 map_uv = vec2(atan(normal_body.y, normal_body.x) / (2.0 * PI) + 0.5,
			0.5 - asin(clamp(normal_body.z, -1.0, 1.0)) / PI);
		surface = texture(surface_map, map_uv).rgb;
	}
	ALBEDO = surface * intensity * (disk + halo);
}
