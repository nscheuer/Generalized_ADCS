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
        self.background.setColor(0.001, 0.0015, 0.0025, 0.72)
        self.background.setTransparency(True)
        self.background.setDepthTest(False)
        self.background.setDepthWrite(False)
        self.text = OnscreenText(text='1000.0 FPS  |  1000.0 ms/frame', pos=(0, 0), scale=0.027,
                                 align=TextNode.ARight, fg=(0.79, 0.90, 0.98, 1),
                                 font=font, mayChange=True, parent=base.aspect2d)
        self._max_text_width = self.text.textNode.getWidth()
        self.text.setText('-- FPS  |  -- ms/frame')
        self.resize()

    def resize(self) -> None:
        left_bound, right_bound = self.base.a2dLeft, self.base.a2dRight
        bottom_bound, top_bound = self.base.a2dBottom, self.base.a2dTop
        ui_scale = max(1.0, top_bound)
        margin_x = min(0.055 * ui_scale, (right_bound - left_bound) * 0.04)
        margin_top = min(0.035 * ui_scale, (top_bound - bottom_bound) * 0.025)
        available_width = max(0.1, right_bound - left_bound - 2 * margin_x)
        text_scale = min(0.027 * ui_scale,
                         available_width / max(self._max_text_width, 1e-6))
        text_width = self._max_text_width * text_scale
        panel_width = min(available_width, text_width + 0.07 * ui_scale)
        panel_height = 0.075 * ui_scale
        panel_top = top_bound - margin_top
        panel_bottom = panel_top - panel_height

        self.text.setScale(text_scale)
        self.text.setPos(right_bound - margin_x - 0.035 * ui_scale,
                         panel_bottom + 0.022 * ui_scale)
        self.background.setScale(panel_width / 0.62, 1, panel_height / 0.075)
        self.background.setPos(right_bound - margin_x - panel_width,
                               0, panel_bottom)

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
            self.resize()
            self.elapsed = 0.0

    def destroy(self) -> None:
        self.text.destroy()
        self.background.removeNode()
