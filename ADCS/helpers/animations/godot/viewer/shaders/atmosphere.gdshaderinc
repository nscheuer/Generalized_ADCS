// Rayleigh/Mie scattering and exponential density following NVIDIA GPU Gems 2,
// chapter 16 (Sean O'Neil). Optional transmittance and multiple-scattering
// tables use Hillaire (2020) and Bruneton (2017) when they are built locally.
//
// Everything is in the body's object space (ITRS metres for Earth). Radiance is
// per unit solar irradiance; multiply by PI to get the viewer's display units,
// where a white Lambertian surface under the full Sun reads 1.

uniform sampler2D transmittance_lut : filter_linear, repeat_disable;
uniform sampler2D multiscatter_lut : filter_linear, repeat_disable;
uniform bool has_atmosphere = false;
uniform bool use_atmosphere_lut = false;
uniform float atm_bottom = 6378137.0;
uniform float atm_top = 6538137.0;
uniform float scattering_gain = 1.0;
uniform vec3 rayleigh_scattering = vec3(5.802e-6, 13.558e-6, 33.1e-6);
uniform float rayleigh_h = 10500.0;
uniform vec3 mie_scattering = vec3(3.996e-6);
uniform vec3 mie_extinction = vec3(4.44e-6);
uniform float mie_h = 1450.0;
uniform float mie_g = 0.8;
uniform vec3 ozone_absorption = vec3(0.65e-6, 1.881e-6, 0.085e-6);
uniform float ozone_center = 25000.0;
uniform float ozone_half_width = 15000.0;
uniform int atm_steps = 24;
// Camera position in object space, computed in float64 on the CPU.
uniform vec3 cam_local = vec3(0.0, 0.0, 2.0e7);

vec2 atm_transmittance_uv(float r, float mu) {
	vec2 size = vec2(textureSize(transmittance_lut, 0));
	float H = sqrt(atm_top * atm_top - atm_bottom * atm_bottom);
	float rho = sqrt(max(r * r - atm_bottom * atm_bottom, 0.0));
	float d = max(-r * mu + sqrt(max(r * r * (mu * mu - 1.0) + atm_top * atm_top, 0.0)), 0.0);
	float d_min = atm_top - r;
	float d_max = rho + H;
	float x_mu = (d - d_min) / max(d_max - d_min, 1e-3);
	float x_r = rho / H;
	return vec2(0.5 / size.x + x_mu * (1.0 - 1.0 / size.x), 0.5 / size.y + x_r * (1.0 - 1.0 / size.y));
}

// Approximate optical depth without a lookup table. The curvature term keeps
// the horizon finite, while the signed branch accounts for rays dipping toward
// the planet before leaving the atmosphere (views from orbit).
float atm_column(float r, float mu, float scale_h) {
	float closest_r = r * sqrt(max(1.0 - mu * mu, 0.0));
	float h = max((mu < 0.0 ? closest_r : r) - atm_bottom, 0.0);
	float secant = 1.0 / sqrt(mu * mu + scale_h / atm_bottom);
	return scale_h * exp(-h / scale_h) * secant * (mu < 0.0 ? 2.0 : 1.0);
}

// Transmittance from radius r to the top of the atmosphere along zenith cosine mu.
vec3 atm_transmittance(float r, float mu) {
	if (use_atmosphere_lut) {
		return textureLod(transmittance_lut, atm_transmittance_uv(min(r, atm_top), mu), 0.0).rgb;
	}
	float rayleigh_depth = atm_column(r, mu, rayleigh_h);
	float mie_depth = atm_column(r, mu, mie_h);
	float h = max(r - atm_bottom, 0.0);
	float ozone_fraction = clamp((ozone_center + ozone_half_width - h) / (2.0 * ozone_half_width), 0.0, 1.0);
	float ozone_depth = ozone_half_width * ozone_fraction / sqrt(mu * mu + ozone_half_width / atm_bottom);
	return exp(-(rayleigh_scattering * rayleigh_depth + mie_extinction * mie_depth + ozone_absorption * ozone_depth));
}

// Sun visible (not behind the Earth) from radius r at sun zenith cosine mu_s.
float atm_sun_visible(float r, float mu_s) {
	return mu_s > -sqrt(max(1.0 - (atm_bottom / r) * (atm_bottom / r), 0.0)) ? 1.0 : 0.0;
}

vec3 atm_multiscatter(float r, float mu_s) {
	if (!use_atmosphere_lut) {
		return vec3(0.0);
	}
	vec2 uv = vec2(clamp(mu_s * 0.5 + 0.5, 0.0, 1.0),
		clamp((r - atm_bottom) / (atm_top - atm_bottom), 0.0, 1.0));
	return textureLod(multiscatter_lut, uv, 0.0).rgb;
}

// Entry/exit distances of a ray (origin o, unit direction d) with a sphere of
// radius R at the origin, via the closest approach (stable in float32 for a
// camera far away). Returns (-1, -1) if missed.
vec2 atm_ray_sphere(vec3 o, vec3 d, float R) {
	float b = -dot(o, d);
	vec3 closest = o + d * b;
	float c2 = dot(closest, closest);
	if (c2 > R * R) {
		return vec2(-1.0);
	}
	float h = sqrt(R * R - c2);
	return vec2(b - h, b + h);
}

float atm_rayleigh_phase(float c) {
	return 3.0 / (16.0 * PI) * (1.0 + c * c);
}

float atm_mie_phase(float c) {
	float g = mie_g;
	float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
	return k * (1.0 + c * c) / pow(1.0 + g * g - 2.0 * g * c, 1.5);  // Cornette-Shanks
}

// In-scattered radiance L and transmittance T along o + d t, t in [t0, t1].
void atm_integrate_n(vec3 o, vec3 d, float t0, float t1, vec3 sun, int steps, out vec3 L, out vec3 T) {
	L = vec3(0.0);
	T = vec3(1.0);
	float cos_theta = dot(d, sun);
	float pr = atm_rayleigh_phase(cos_theta);
	float pm = atm_mie_phase(cos_theta);
	float dt = (t1 - t0) / float(steps);
	for (int i = 0; i < steps; i++) {
		vec3 x = o + d * (t0 + (float(i) + 0.5) * dt);
		float r = length(x);
		float h = r - atm_bottom;
		float rho_r = exp(-max(h, 0.0) / rayleigh_h);
		float rho_m = exp(-max(h, 0.0) / mie_h);
		float rho_o = max(0.0, 1.0 - abs(h - ozone_center) / ozone_half_width);
		vec3 s_r = rayleigh_scattering * rho_r;
		vec3 s_m = mie_scattering * rho_m;
		vec3 ext = s_r + mie_extinction * rho_m + ozone_absorption * rho_o;
		float mu_s = dot(x, sun) / r;
		vec3 sun_t = atm_transmittance(r, mu_s) * atm_sun_visible(r, mu_s);
		vec3 S = (s_r * pr + s_m * pm) * sun_t + (s_r + s_m) * atm_multiscatter(r, mu_s);
		vec3 step_t = exp(-ext * dt);
		L += T * S * (1.0 - step_t) / max(ext, vec3(1e-12));  // exact for constant S over the step
		T *= step_t;
	}
	L *= scattering_gain;
}

void atm_integrate(vec3 o, vec3 d, float t0, float t1, vec3 sun, out vec3 L, out vec3 T) {
	atm_integrate_n(o, d, t0, t1, sun, atm_steps, L, T);
}

// Atmosphere between the camera and surface point p (object space).
void atm_aerial(vec3 p, vec3 sun, int steps, out vec3 L, out vec3 T) {
	vec3 d = normalize(p - cam_local);
	vec2 top = atm_ray_sphere(cam_local, d, atm_top);
	atm_integrate_n(cam_local, d, max(top.x, 0.0), length(p - cam_local), sun, steps, L, T);
}

