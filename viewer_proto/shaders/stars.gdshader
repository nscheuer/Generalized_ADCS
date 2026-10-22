// Catalog-like star points at a fixed large distance.
shader_type spatial;
render_mode unshaded, cull_disabled, blend_add;

void vertex() {
	POINT_SIZE = 2.0;
}

void fragment() {
	ALBEDO = COLOR.rgb;
}
