// Catalogue stars as points. COLOR.rgb is the chromaticity, UV.x the
// illuminance relative to a V = 0 star.
//  - artistic: compressed brightness and sizes, visible next to a lit Earth;
//  - physical: each star's flux into one pixel (phys_gain = pi E_V0 / (omega_px E_sun)),
//    so with an exposure set for a sunlit Earth they vanish, as for a real camera.
shader_type spatial;
render_mode unshaded, cull_disabled, blend_add, depth_draw_never;

uniform bool physical = false;
uniform float phys_gain = 1.0e-3;
uniform float artistic_gain = 1.0;

varying vec3 chroma;
varying float e_rel;

void vertex() {
	chroma = COLOR.rgb;
	e_rel = UV.x;
	POINT_SIZE = physical ? 1.0 : clamp(1.0 + clamp(pow(e_rel, 0.4), 0.03, 4.0), 1.0, 4.0);
}

void fragment() {
	float b = physical ? e_rel * phys_gain : clamp(pow(e_rel, 0.4), 0.03, 4.0) * artistic_gain;
	ALBEDO = chroma * b;
}
