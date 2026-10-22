"""Sky sphere using NASA's Deep Star Maps 2020 textures."""

from ._geometry import uv_sphere
from ._texture_assets import texture_path


class Starfield:
    """Render the selected equirectangular star map behind the scene."""

    def __init__(self, base, tier="medium") -> None:
        from panda3d.core import TextureStage

        self.base = base
        self.texture = None
        self.set_quality(tier)
        self.root = base.render.attachNewNode('inertial starfield')
        self.mesh = self.root.attachNewNode(uv_sphere(1.0, slices=96, stacks=48))
        self.mesh.setTexture(self.texture, 1)
        stage = TextureStage.getDefault()
        self.mesh.setTexScale(stage, -1, 1)
        self.mesh.setTexOffset(stage, 1, 0)
        self.mesh.setColor(1, 1, 1, 1)
        self.mesh.setLightOff()
        self.mesh.setTwoSided(True)
        self.mesh.setDepthTest(False)
        self.mesh.setDepthWrite(False)
        self.mesh.setBin('background', 0)
        self._radius = None

    def set_quality(self, tier: str) -> None:
        from panda3d.core import Texture
        name = 'starmap_2020_16k.exr' if tier == 'high' else 'starmap_2020_4k.exr'
        path = texture_path(name)
        new_texture = self.base.loader.loadTexture(str(path))
        if new_texture is None:
            raise RuntimeError(f'Could not load starfield texture: {path}')
        old_texture = self.texture
        self.texture = new_texture
        # A single bilinear lookup keeps the HDR star map inexpensive to render.
        self.texture.setMinfilter(Texture.FTLinear)
        self.texture.setMagfilter(Texture.FTLinear)
        self.texture.setAnisotropicDegree(1)
        self.texture.setWrapU(Texture.WMRepeat)
        self.texture.setWrapV(Texture.WMClamp)
        self.texture.setCompression(Texture.CM_off)
        self.texture.setKeepRamImage(False)

        if hasattr(self, 'mesh'):
            self.mesh.setTexture(new_texture, 1)
        if old_texture is not None:
            self.base.loader.unloadTexture(old_texture)

    def update(self, source_camera, camera_distance: float) -> None:
        """Follow camera translation without inheriting camera rotation."""
        self.root.setPos(source_camera.getPos(self.base.render))
        radius = max(10.0, float(camera_distance) * 80.0)
        if radius != self._radius:
            self.root.setScale(radius)
            self._radius = radius

    def close(self) -> None:
        self.root.removeNode()
        self.base.loader.unloadTexture(self.texture)
