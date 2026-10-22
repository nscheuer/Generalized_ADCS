// Catalogue stars as points; COLOR.rgb is linear radiance-like brightness,
// COLOR.a the point size in pixels.
shader_type spatial;
render_mode unshaded, cull_disabled, blend_add, depth_draw_never;

uniform float gain = 1.0;

void vertex() {
	POINT_SIZE = COLOR.a;
}

void fragment() {
	ALBEDO = COLOR.rgb * gain;
}
