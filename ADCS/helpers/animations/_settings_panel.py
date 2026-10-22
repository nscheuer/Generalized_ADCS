"""Compact, extensible screen-space viewer settings panel."""


class SettingsPanel:
    """A collapsible right-side panel with callback-backed toggle settings."""

    BUTTON_WIDTH = 0.76
    BUTTON_HEIGHT = 0.09
    ROW_HEIGHT = 0.115
    PANEL_PADDING = 0.045

    def __init__(self, base, font) -> None:
        self.base = base
        self.font = font
        self.open = False
        self.options = []
        self._nodes = []
        self._texts = []
        self._bounds = {}
        self.resize()

    def add_toggle(self, key: str, label: str, get_value, set_value) -> None:
        if any(option[0] == key for option in self.options):
            raise ValueError(f"Settings option already exists: {key}")
        self.options.append((key, label, get_value, set_value))
        self.resize()

    def _card(self, name, left, right, bottom, top, color, *, fixed_bin=20):
        from panda3d.core import CardMaker
        maker = CardMaker(name)
        maker.setFrame(left, right, bottom, top)
        node = self.base.aspect2d.attachNewNode(maker.generate())
        node.setColor(*color)
        node.setTransparency(True)
        node.setDepthTest(False)
        node.setDepthWrite(False)
        node.setBin("fixed", fixed_bin)
        self._nodes.append(node)
        return node

    def _text(self, value, x, z, scale, color, align):
        from direct.gui.OnscreenText import OnscreenText
        label = OnscreenText(
            text=value, pos=(x, z), scale=scale * self.ui_scale, align=align,
            fg=color, font=self.font, mayChange=True, parent=self.base.aspect2d,
        )
        label.setDepthTest(False)
        label.setDepthWrite(False)
        label.setBin("fixed", 21)
        self._texts.append(label)
        return label

    def resize(self) -> None:
        for node in self._nodes:
            node.removeNode()
        for label in self._texts:
            label.destroy()
        self._nodes.clear()
        self._texts.clear()
        self._bounds.clear()

        self.ui_scale = max(1.0, self.base.a2dTop)
        right = self.base.a2dRight - 0.055 * self.ui_scale
        top = self.base.a2dTop - 0.13 * self.ui_scale
        width = min(self.BUTTON_WIDTH * self.ui_scale,
                    max(0.48 * self.ui_scale, (self.base.a2dRight - self.base.a2dLeft) * 0.40))
        height = self.BUTTON_HEIGHT * self.ui_scale
        self._card("viewer settings button", right - width, right,
                   top - height, top, (0.008, 0.012, 0.018, 0.92))
        self._text("SETTINGS  +" if not self.open else "SETTINGS  -",
                   right - 0.035 * self.ui_scale, top - 0.065 * self.ui_scale,
                   0.024, (0.77, 0.88, 0.97, 1), 2)
        self._bounds["button"] = (right - width, right, top - height, top)

        if not self.open:
            return
        panel_height = self.PANEL_PADDING * 2 * self.ui_scale + len(self.options) * self.ROW_HEIGHT * self.ui_scale
        panel_top = top - height - 0.015 * self.ui_scale
        panel_bottom = panel_top - panel_height
        self._card("viewer settings panel", right - width, right,
                   panel_bottom, panel_top, (0.006, 0.009, 0.014, 0.97))
        self._bounds["panel"] = (right - width, right, panel_bottom, panel_top)
        for index, (key, label, get_value, _set_value) in enumerate(self.options):
            row_top = panel_top - self.PANEL_PADDING * self.ui_scale - index * self.ROW_HEIGHT * self.ui_scale
            row_bottom = row_top - self.ROW_HEIGHT * self.ui_scale
            checked = bool(get_value())
            self._card(f"setting {key} checkbox", right - width + 0.035 * self.ui_scale,
                       right - width + 0.087 * self.ui_scale,
                       row_bottom + 0.031 * self.ui_scale,
                       row_bottom + 0.083 * self.ui_scale,
                       (0.10, 0.58, 0.83, 1) if checked else (0.06, 0.075, 0.09, 1),
                       fixed_bin=22)
            if checked:
                # Panda's default font does not include U+2713 on all systems.
                self._text("x", right - width + 0.061 * self.ui_scale,
                           row_bottom + 0.036 * self.ui_scale, 0.020,
                           (0.96, 0.99, 1, 1), 1)
            self._text(label, right - width + 0.115 * self.ui_scale,
                       row_bottom + 0.038 * self.ui_scale, 0.023,
                       (0.87, 0.91, 0.96, 1), 0)
            self._bounds[key] = (right - width, right, row_bottom, row_top)

    def handle_pointer(self, mouse) -> bool:
        if mouse is None:
            return False
        x = float(mouse.x) * self.base.a2dRight
        z = float(mouse.y) * self.base.a2dTop
        button = self._bounds.get("button")
        if button and button[0] <= x <= button[1] and button[2] <= z <= button[3]:
            self.open = not self.open
            self.resize()
            return True
        panel = self._bounds.get("panel")
        if panel and panel[0] <= x <= panel[1] and panel[2] <= z <= panel[3]:
            for key, _label, get_value, set_value in self.options:
                bounds = self._bounds[key]
                if bounds[2] <= z <= bounds[3]:
                    set_value(not bool(get_value()))
                    self.resize()
                    break
            return True
        return False

    def close(self) -> None:
        for node in self._nodes:
            node.removeNode()
        for label in self._texts:
            label.destroy()
        self._nodes.clear()
        self._texts.clear()
