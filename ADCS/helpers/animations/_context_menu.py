"""Small screen-space context menu for celestial scene objects."""


class BodyContextMenu:
    """A lightweight menu with one action, positioned at the pointer."""

    LABEL = 'Focus on object'

    def __init__(self, base, font, on_focus) -> None:
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import CardMaker, TextNode

        self.base = base
        self.on_focus = on_focus
        self.name = None
        self.visible = False
        self._scale = 1.0
        self._width = 0.58
        self._height = 0.12

        maker = CardMaker('body context menu')
        maker.setFrame(0, self._width, -self._height, 0)
        self.frame = base.aspect2d.attachNewNode(maker.generate())
        self.frame.setColor(0.012, 0.016, 0.023, 1)
        self.frame.setDepthTest(False)
        self.frame.setDepthWrite(False)
        self.frame.setBin('fixed', 30)
        self.frame.hide()

        self.label = OnscreenText(
            text=self.LABEL,
            pos=(0.035, -0.079),
            scale=0.030,
            align=TextNode.ALeft,
            fg=(0.88, 0.94, 1.0, 1),
            font=font,
            mayChange=False,
            parent=self.frame,
        )
        self.label.setDepthTest(False)
        self.label.setDepthWrite(False)
        self.label.setBin('fixed', 31)

        self.resize()

    def resize(self) -> None:
        self._scale = max(1.0, self.base.a2dTop)
        self.frame.setScale(self._scale)

    def show(self, name: str, mouse) -> None:
        self.name = name
        self.visible = True
        self.resize()
        aspect = self.base.a2dRight
        left = float(mouse.x) * aspect
        top = float(mouse.y) * self.base.a2dTop
        width = self._width * self._scale
        height = self._height * self._scale
        left = min(max(self.base.a2dLeft + 0.02, left),
                   self.base.a2dRight - width - 0.02)
        top = min(max(self.base.a2dBottom + height + 0.02, top),
                  self.base.a2dTop - 0.02)
        self.frame.setPos(left, 0, top)
        self.frame.show()

    def contains(self, mouse) -> bool:
        if not self.visible or mouse is None:
            return False
        left = self.frame.getX()
        top = self.frame.getZ()
        mouse_y = mouse.y * self.base.a2dTop
        return (left <= mouse.x * self.base.a2dRight <= left + self._width * self._scale
                and top - self._height * self._scale <= mouse_y <= top)

    def activate(self, mouse) -> bool:
        if not self.contains(mouse):
            return False
        name = self.name
        self.hide()
        self.on_focus(name)
        return True

    def hide(self) -> None:
        self.visible = False
        self.name = None
        self.frame.hide()

    def close(self) -> None:
        self.label.destroy()
        self.frame.removeNode()
