"""Low-overhead live frame-rate and frame-time overlay."""

from math import exp


class PerformanceOverlay:
    def __init__(self, base, font) -> None:
        self.base = base
        self.font = font
        self.average_dt = None
        self.elapsed = 0.0
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import CardMaker, TextNode

        maker = CardMaker('performance HUD background')
        maker.setFrame(0, 0.62, 0, 0.075)
        self.background = base.aspect2d.attachNewNode(maker.generate())
        self.background.setColor(0.012, 0.025, 0.043, 0.78)
        self.background.setTransparency(True)
        self.background.setDepthTest(False)
        self.background.setDepthWrite(False)
        self.text = OnscreenText(text='-- FPS  |  -- ms/frame', pos=(0, 0), scale=0.027,
                                 align=TextNode.ARight, fg=(0.79, 0.90, 0.98, 1),
                                 font=font, mayChange=True, parent=base.aspect2d)
        self.resize()

    def resize(self) -> None:
        aspect = self.base.getAspectRatio()
        right = aspect - 0.055
        self.background.setPos(right - 0.64, 0, 0.89)
        self.text.setPos(right - 0.035, 0.912)

    def update(self, frame_dt: float) -> None:
        if frame_dt > 0:
            if self.average_dt is None:
                self.average_dt = frame_dt
            else:
                alpha = 1.0 - exp(-frame_dt / 0.5)
                self.average_dt += alpha * (frame_dt - self.average_dt)
        self.elapsed += max(0.0, frame_dt)
        if self.elapsed >= 0.15 and self.average_dt:
            fps = 1.0 / self.average_dt
            ms = self.average_dt * 1000.0
            self.text.setText(f'{fps:5.1f} FPS  |  {ms:5.1f} ms/frame')
            self.elapsed = 0.0

    def destroy(self) -> None:
        self.text.destroy()
        self.background.removeNode()
