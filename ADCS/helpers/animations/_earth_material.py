"""GPU Earth material using the repository's day, night, cloud and surface maps."""

from pathlib import Path
from ._texture_assets import texture_path, tier_name

TEXTURE_DIR = Path(__file__).resolve().parents[1] / "plotting" / "textures"
EARTH_MAPS = {
    "day_map": "2k_earth_daymap.jpg",
    "night_map": "2k_earth_nightmap.jpg",
    "cloud_map": "2k_earth_clouds.jpg",
    "normal_map": "2k_earth_normal_map.tif",
    "specular_map": "2k_earth_specular_map.tif",
}

VERTEX_SHADER = r"""#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
attribute vec4 p3d_Vertex;
attribute vec3 p3d_Normal;
attribute vec2 p3d_MultiTexCoord0;
varying vec3 local_position;
varying vec3 local_normal;
varying vec2 earth_uv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    local_position = p3d_Vertex.xyz;
    local_normal = p3d_Normal;
    earth_uv = p3d_MultiTexCoord0;
}
"""

FRAGMENT_SHADER = r"""#version 120
uniform sampler2D day_map;
uniform sampler2D night_map;
uniform sampler2D cloud_map;
uniform sampler2D normal_map;
uniform sampler2D specular_map;
uniform sampler2D low_day_map;
uniform sampler2D low_night_map;
uniform sampler2D low_cloud_map;
uniform sampler2D low_normal_map;
uniform sampler2D low_specular_map;
uniform vec3 sun_direction;
uniform vec3 camera_position;
uniform float cloud_offset;
uniform float normal_strength;
uniform float texture_blend;
uniform float detail_strength;
uniform float resolution_blend;
varying vec3 local_position;
varying vec3 local_normal;
varying vec2 earth_uv;
void main() {
    vec3 geometric_normal = normalize(local_normal);
    vec3 p = normalize(local_position);
    vec3 dp1 = dFdx(p);
    vec3 dp2 = dFdy(p);
    vec2 duv1 = dFdx(earth_uv);
    vec2 duv2 = dFdy(earth_uv);
    float det = duv1.x * duv2.y - duv1.y * duv2.x;
    vec3 tangent = normalize((dp1 * duv2.y - dp2 * duv1.y) / max(abs(det), 0.00001));
    vec3 bitangent = normalize((-dp1 * duv2.x + dp2 * duv1.x) / max(abs(det), 0.00001));
    vec3 surface_normal = geometric_normal;
    if (normal_strength > 0.0) {
        vec3 mapped = texture2D(normal_map, earth_uv).xyz;
        if (resolution_blend < 1.0)
            mapped = mix(texture2D(low_normal_map, earth_uv).xyz, mapped, resolution_blend);
        mapped = mapped * 2.0 - 1.0;
        surface_normal = normalize(mat3(tangent, bitangent, geometric_normal) * mapped);
        surface_normal = normalize(mix(geometric_normal, surface_normal, normal_strength));
    }

    vec3 light_dir = normalize(sun_direction);
    float illumination = dot(surface_normal, light_dir);
    float day = smoothstep(-0.10, 0.12, illumination);
    vec3 day_color = texture2D(day_map, earth_uv).rgb;
    vec3 night_color = texture2D(night_map, earth_uv).rgb * 1.25;
    if (resolution_blend < 1.0) {
        day_color = mix(texture2D(low_day_map, earth_uv).rgb, day_color, resolution_blend);
        night_color = mix(texture2D(low_night_map, earth_uv).rgb * 1.25, night_color, resolution_blend);
    }
    vec3 color = mix(night_color, day_color * (0.08 + 0.92 * max(illumination, 0.0)), day);
    vec3 base_color = vec3(0.33, 0.55, 0.83) * (0.08 + 0.92 * max(dot(geometric_normal, light_dir), 0.0));
    color = mix(base_color, color, texture_blend);

    vec2 cloud_uv = vec2(fract(earth_uv.x + cloud_offset), earth_uv.y);
    float cloud_value = dot(texture2D(cloud_map, cloud_uv).rgb, vec3(0.299, 0.587, 0.114));
    if (resolution_blend < 1.0)
        cloud_value = mix(dot(texture2D(low_cloud_map, cloud_uv).rgb, vec3(0.299, 0.587, 0.114)), cloud_value, resolution_blend);
    float clouds = smoothstep(0.34, 0.76, cloud_value);
    float cloud_light = mix(0.34, 0.88, smoothstep(-0.28, 0.18, illumination));
    color = mix(color, vec3(0.88, 0.93, 1.0) * cloud_light, clouds * 0.48 * detail_strength);

    vec3 view_dir = normalize(camera_position - local_position);
    vec3 half_dir = normalize(light_dir + view_dir);
    float spec_mask = texture2D(specular_map, earth_uv).r;
    if (resolution_blend < 1.0)
        spec_mask = mix(texture2D(low_specular_map, earth_uv).r, spec_mask, resolution_blend);
    float specular = pow(max(dot(surface_normal, half_dir), 0.0), 52.0) * spec_mask;
    color += vec3(0.48, 0.60, 0.82) * specular * day * detail_strength;

    float rim = pow(1.0 - max(dot(geometric_normal, view_dir), 0.0), 3.0);
    color += vec3(0.018, 0.10, 0.24) * rim * (0.22 + 0.78 * day) * detail_strength;
    gl_FragColor = vec4(color, 1.0);
}
"""

SIMPLE_FRAGMENT_SHADER = r"""#version 120
uniform sampler2D day_map;
uniform vec3 sun_direction;
uniform float texture_blend;
varying vec3 local_normal;
varying vec2 earth_uv;
void main() {
    float light = 0.08 + 0.92 * max(dot(normalize(local_normal), normalize(sun_direction)), 0.0);
    vec3 base_color = vec3(0.33, 0.55, 0.83);
    vec3 color = mix(base_color, texture2D(day_map, earth_uv).rgb, texture_blend);
    gl_FragColor = vec4(color * light, 1.0);
}
"""


class EarthMaterial:
    """Own and bind Earth's five GPU maps and animated shader inputs."""

    def __init__(self, base, node, day_texture_path=None, tier="medium") -> None:
        from panda3d.core import Shader

        self.base = base
        self.node = node
        self.day_texture_path = day_texture_path
        self.tier = tier
        self.paths = self._paths_for(tier)
        for path in self.paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Earth texture not found: {path}")
        self.textures = {}
        self.blend_textures = {}
        self.shader = Shader.make(Shader.SL_GLSL, VERTEX_SHADER, FRAGMENT_SHADER)
        if self.shader is None:
            raise RuntimeError("Panda3D could not create the Earth shader")
        self.sun_direction = (0.0, 0.0, 1.0)
        self.camera_position = (0.0, -3.0, 0.0)
        self.cloud_offset = 0.0
        self.normal_strength = 0.0
        self.texture_blend = 1.0
        self.detail_strength = 1.0
        self.resolution_blend = 1.0
        self.enabled = False

    def _paths_for(self, tier):
        paths = {key: texture_path(tier_name(tier, filename))
                 for key, filename in EARTH_MAPS.items()}
        if self.day_texture_path is not None:
            paths["day_map"] = Path(self.day_texture_path).expanduser()
        return paths

    def set_quality(self, tier):
        if tier == self.tier:
            return
        was_enabled = self.enabled
        if was_enabled:
            self.release()
        self.tier = tier
        self.paths = self._paths_for(tier)
        if was_enabled:
            self.set_enabled(True)

    def set_enabled(self, enabled: bool) -> None:
        if enabled and not self.enabled:
            self.node.setShader(self.shader, 1)
            for name, path in self.paths.items():
                texture = self.base.loader.loadTexture(str(path))
                if texture is None:
                    self.release()
                    raise RuntimeError(f"Could not load Earth map: {path}")
                texture.setMinfilter(texture.FTLinearMipmapLinear)
                texture.setMagfilter(texture.FTLinear)
                texture.setWrapU(texture.WMRepeat)
                texture.setWrapV(texture.WMClamp)
                self.textures[name] = texture
                self.node.setShaderInput(name, texture)
                if self.tier != 'high':
                    self.node.setShaderInput('low_' + name, texture)
            if self.tier == 'high':
                for name, path in self._paths_for('medium').items():
                    texture = self.base.loader.loadTexture(str(path))
                    if texture is None:
                        self.release()
                        raise RuntimeError(f"Could not load Earth map: {path}")
                    texture.setMinfilter(texture.FTLinearMipmapLinear)
                    texture.setMagfilter(texture.FTLinear)
                    texture.setWrapU(texture.WMRepeat)
                    texture.setWrapV(texture.WMClamp)
                    self.blend_textures[name] = texture
                    self.node.setShaderInput('low_' + name, texture)
            self.enabled = True
            self._apply_inputs()
        elif not enabled and self.enabled:
            self.release()

    def set_sun_direction(self, vector) -> None:
        self.sun_direction = tuple(float(v) for v in vector)
        self._apply_inputs()

    def set_camera_position(self, vector) -> None:
        self.camera_position = tuple(float(v) for v in vector)
        self._apply_inputs()

    def set_cloud_offset(self, offset: float) -> None:
        self.cloud_offset = float(offset) % 1.0
        self._apply_inputs()

    def set_normal_strength(self, strength: float) -> None:
        self.normal_strength = float(strength)
        self._apply_inputs()

    def set_lod_blend(self, texture_blend: float, detail_strength: float,
                      resolution_blend: float = 1.0) -> None:
        self.texture_blend = float(texture_blend)
        self.detail_strength = float(detail_strength)
        self.resolution_blend = float(resolution_blend)
        self._apply_inputs()

    def _apply_inputs(self) -> None:
        if not self.enabled:
            return
        self.node.setShaderInput("sun_direction", *self.sun_direction)
        self.node.setShaderInput("camera_position", *self.camera_position)
        self.node.setShaderInput("cloud_offset", self.cloud_offset)
        self.node.setShaderInput("normal_strength", self.normal_strength)
        self.node.setShaderInput("texture_blend", self.texture_blend)
        self.node.setShaderInput("detail_strength", self.detail_strength)
        self.node.setShaderInput("resolution_blend", self.resolution_blend)

    def release(self) -> None:
        for name in (*self.paths.keys(), *(f'low_{name}' for name in EARTH_MAPS), "sun_direction", "camera_position", "cloud_offset", "normal_strength", "texture_blend", "detail_strength", "resolution_blend"):
            self.node.clearShaderInput(name)
        self.node.clearShader()
        for texture in self.textures.values():
            self.base.loader.unloadTexture(texture)
        for texture in self.blend_textures.values():
            self.base.loader.unloadTexture(texture)
        self.textures.clear()
        self.blend_textures.clear()
        self.enabled = False

    def close(self) -> None:
        self.release()
