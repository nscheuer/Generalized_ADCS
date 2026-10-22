"""Single-pass, sunlit atmosphere shell for the high quality Earth view."""

from ._geometry import uv_sphere


VERTEX_SHADER = """#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
attribute vec4 p3d_Vertex;
varying vec3 position;
void main() {
    position = p3d_Vertex.xyz;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""

FRAGMENT_SHADER = """#version 120
uniform vec3 sun_direction;
uniform vec3 camera_position;
uniform float intensity;
uniform float planet_radius;
varying vec3 position;
void main() {
    float ground = planet_radius;
    float outer = ground * 1.06;
    float scale_height = ground * 0.014;
    vec3 ray = normalize(position - camera_position);
    vec3 sun = normalize(sun_direction);

    // Intersect the view ray with the air sphere and stop at solid ground.
    float b = dot(camera_position, ray);
    float c = dot(camera_position, camera_position) - outer * outer;
    float root = sqrt(max(0.0, b * b - c));
    float start = max(0.0, -b - root);
    float end = -b + root;
    float ground_c = dot(camera_position, camera_position) - ground * ground;
    float ground_discriminant = b * b - ground_c;
    if (ground_discriminant > 0.0) {
        float ground_hit = -b - sqrt(ground_discriminant);
        if (ground_hit > start) end = min(end, ground_hit);
    }
    if (end <= start) discard;

    float step_length = (end - start) / 8.0;
    float optical_depth = 0.0;
    float lit_depth = 0.0;
    vec3 in_scatter = vec3(0.0);
    for (int i = 0; i < 8; ++i) {
        vec3 sample_point = camera_position + ray * (start + (float(i) + 0.5) * step_length);
        float radius = length(sample_point);
        float density = exp(-max(0.0, radius - ground) / scale_height);
        float segment = density * step_length / ground;
        optical_depth += segment;

        // Sun rays crossing solid Earth cast a sharp planetary shadow.
        float sun_mu = dot(sample_point, sun) / radius;
        float impact = radius * sqrt(max(0.0, 1.0 - sun_mu * sun_mu));
        float sunlit = (sun_mu >= 0.0 || impact >= ground) ? 1.0 : 0.0;
        float sunlight_path = max(0.0, outer - radius) / max(0.10, sun_mu);
        float sunlight = sunlit * exp(-density * sunlight_path * 2.8 / ground);
        float contribution = segment * sunlight * exp(-optical_depth * 2.8);
        lit_depth += contribution;

        // Rayleigh scattering favors blue; shallow sun angles warm the rim.
        vec3 blue = vec3(0.23, 0.48, 1.0);
        vec3 sunset = vec3(1.0, 0.48, 0.20);
        vec3 tint = mix(sunset, blue, smoothstep(-0.03, 0.28, sun_mu));
        in_scatter += tint * contribution;
    }
    float phase = 0.75 * (1.0 + pow(dot(ray, sun), 2.0));
    float alpha = intensity * (1.0 - exp(-lit_depth * 4.5)) * phase;
    vec3 color = in_scatter / max(lit_depth, 0.00001);
    gl_FragColor = vec4(color, clamp(alpha, 0.0, 0.85));
}
"""


class EarthAtmosphere:
    def __init__(self, root, radius):
        from panda3d.core import Shader, TransparencyAttrib

        self.node = root.attachNewNode(uv_sphere(radius * 1.06, slices=64, stacks=32))
        self.node.setShader(Shader.make(Shader.SL_GLSL, VERTEX_SHADER, FRAGMENT_SHADER))
        self.node.setShaderInput('planet_radius', radius)
        self.node.setShaderInput('sun_direction', 0.0, 0.0, 1.0)
        self.node.setShaderInput('camera_position', 0.0, -radius * 5.0, 0.0)
        self.node.setShaderInput('intensity', 0.0)
        self.node.setTransparency(TransparencyAttrib.MAlpha)
        self.node.setDepthTest(True)
        self.node.setDepthWrite(False)
        self.node.setBin('transparent', 0)
        self.node.setLightOff()
        self.node.hide()

    def set_sun_direction(self, direction):
        self.node.setShaderInput('sun_direction', *direction)

    def set_camera_position(self, position):
        self.node.setShaderInput('camera_position', *position)

    def set_intensity(self, intensity):
        if intensity > 0.0:
            self.node.setShaderInput('intensity', intensity)
            self.node.show()
        else:
            self.node.hide()

    def close(self):
        self.node.removeNode()
