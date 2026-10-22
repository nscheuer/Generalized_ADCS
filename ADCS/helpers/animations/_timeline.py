"""Interactive, unbounded UTC ruler and track viewport."""

from datetime import datetime, timezone
from math import floor, log10

from ._timeline_model import TimelineConfig, TimelineModel, utc_seconds
from ._fonts import load_ui_font


def _utc_label(timestamp: float, span: float) -> str:
    try:
        dt = datetime.fromtimestamp(timestamp, timezone.utc)
        if span < 2:
            return dt.strftime('%H:%M:%S.') + f'{dt.microsecond // 1000:03d}'
        if span < 86400:
            return dt.strftime('%H:%M:%S')
        return dt.strftime('%d %b %H:%M')
    except (OverflowError, OSError, ValueError):
        return f'{timestamp:.0f}'


def _tick_step(span: float) -> float:
    desired = span / 6
    power = 10 ** floor(log10(desired))
    for multiplier in (1, 2, 5, 10):
        if multiplier * power >= desired:
            return multiplier * power
    return 10 * power


class TimelinePanel:
    """Persistent playback controls plus a movable view into UTC time."""

    def __init__(self, base, playback, model: TimelineModel | None = None,
                 config: TimelineConfig | None = None) -> None:
        self.base = base
        self.font = load_ui_font(base)
        self.playback = playback
        self.model = model if model is not None else TimelineModel()
        self.config = config if config is not None else TimelineConfig()
        interval = playback.data_interval
        self.span = self.config.initial_span_seconds or (max(10.0, (interval[1] - interval[0]) * 4) if interval else 600.0)
        self.span = max(self.config.min_span_seconds, min(self.config.max_span_seconds, self.span))
        self.center = playback.current + self.span * 0.25
        self.follow = True
        self.row_offset = 0
        self._pointer_start = None
        self._pointer_center = None
        self._dragged = False
        self._static_nodes = []
        self._ruler_nodes = []
        self._static_texts = []
        self._ruler_texts = []
        self._readout = None
        self._play_text = None
        self._speed_text = None
        self._playhead = None
        self.resize()

    @property
    def top(self) -> float:
        return -1 + 2 * self.config.height_fraction

    def _card(self, name, left, right, bottom, top, color, *, dynamic=False):
        from panda3d.core import CardMaker
        maker = CardMaker(name)
        maker.setFrame(left, right, bottom, top)
        node = self.base.aspect2d.attachNewNode(maker.generate())
        node.setColor(*color)
        node.setTransparency(True)
        node.setDepthTest(False)
        node.setDepthWrite(False)
        (self._ruler_nodes if dynamic else self._static_nodes).append(node)
        return node

    def _text(self, value, x, y, scale=0.032, color=(0.73, 0.83, 0.93, 1),
              *, dynamic=False, align=None, may_change=False):
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode
        label = OnscreenText(text=value, pos=(x, y), scale=scale,
                             align=align or TextNode.ALeft, fg=color,
                             mayChange=may_change, font=self.font,
                             parent=self.base.aspect2d)
        (self._ruler_texts if dynamic else self._static_texts).append(label)
        return label

    def resize(self) -> None:
        for node in self._static_nodes:
            node.removeNode()
        for label in self._static_texts:
            label.destroy()
        self._static_nodes.clear()
        self._static_texts.clear()
        aspect = self.base.getAspectRatio()
        self.left = -aspect + 0.04
        self.control_right = min(self.left + 1.05, aspect - 0.36)
        self.ruler_left = self.control_right + 0.08
        self.ruler_right = aspect - 0.06
        self.ruler_width = max(0.1, self.ruler_right - self.ruler_left)
        self._card('timeline background', -aspect, aspect, -1, self.top,
                   (0.011, 0.023, 0.043, 1))
        self._card('timeline top rule', -aspect, aspect, self.top - 0.004,
                   self.top + 0.004, (0.13, 0.26, 0.39, 1))
        self._card('timeline controls', self.left, self.control_right,
                   -0.965, self.top - 0.027, (0.020, 0.040, 0.070, 1))
        self._card('timeline ruler well', self.ruler_left - 0.025,
                   self.ruler_right + 0.025, -0.965, self.top - 0.027,
                   (0.020, 0.040, 0.070, 1))
        self._readout = self._text('', self.left + 0.045, self.top - 0.09,
                                   scale=0.036, color=(0.91, 0.97, 1, 1), may_change=True)
        self._text('UTC', self.left + 0.045, self.top - 0.135, 0.025,
                   (0.43, 0.66, 0.81, 1))
        button_y = -0.82
        self._button_centers = (self.left + 0.16, self.left + 0.41, self.left + 0.66)
        for x in self._button_centers:
            self._card('playback button', x - 0.095, x + 0.095,
                       button_y - 0.049, button_y + 0.053,
                       (0.07, 0.13, 0.20, 1))
        from panda3d.core import TextNode
        self._text('-', self._button_centers[0], button_y - 0.012, 0.050,
                   (0.81, 0.91, 1, 1), align=TextNode.ACenter)
        self._play_text = self._text('', self._button_centers[1], button_y - 0.010,
                                     0.039, (0.81, 0.91, 1, 1), align=TextNode.ACenter,
                                     may_change=True)
        self._text('+', self._button_centers[2], button_y - 0.012, 0.047,
                   (0.81, 0.91, 1, 1), align=TextNode.ACenter)
        self._speed_text = self._text('', self.left + 0.80, button_y - 0.012,
                                      0.031, (0.54, 0.84, 0.98, 1), may_change=True)
        self._rebuild_ruler()
        self.set_time(self.playback.current)

    def _clear_ruler(self) -> None:
        for node in self._ruler_nodes:
            node.removeNode()
        for label in self._ruler_texts:
            label.destroy()
        self._ruler_nodes.clear()
        self._ruler_texts.clear()

    def _x_at(self, timestamp: float) -> float:
        return self.ruler_left + (timestamp - (self.center - self.span / 2)) / self.span * self.ruler_width

    def _time_at(self, x: float) -> float:
        return self.center - self.span / 2 + (x - self.ruler_left) / self.ruler_width * self.span

    def _rebuild_ruler(self) -> None:
        self._clear_ruler()
        bottom, top = self.center - self.span / 2, self.center + self.span / 2
        step = _tick_step(self.span)
        tick = floor(bottom / step) * step
        for _ in range(12):
            x = self._x_at(tick)
            if x > self.ruler_right:
                break
            if x >= self.ruler_left:
                self._card('UTC tick', x, x + 0.002, self.top - 0.18,
                           self.top - 0.13, (0.22, 0.39, 0.54, 1), dynamic=True)
                self._text(_utc_label(tick, self.span), x + 0.009,
                           self.top - 0.12, 0.025, dynamic=True)
            minor_step = step / 5
            for minor_index in range(1, 5):
                minor_time = tick + minor_step * minor_index
                minor_x = self._x_at(minor_time)
                if self.ruler_left <= minor_x <= self.ruler_right:
                    self._card('UTC subtick', minor_x, minor_x + 0.0015,
                               self.top - 0.164, self.top - 0.137,
                               (0.10, 0.22, 0.33, 1), dynamic=True)
            tick += step
        self._card('track separator', self.ruler_left, self.ruler_right,
                   self.top - 0.21, self.top - 0.208,
                   (0.10, 0.21, 0.32, 1), dynamic=True)
        for index, (depth, track) in enumerate(self.model.rows()):
            if index < self.row_offset:
                continue
            y = self.top - 0.26 - (index - self.row_offset) * self.config.row_height
            if y - self.config.row_height < -0.96:
                break
            self._card('track stripe', self.ruler_left, self.ruler_right,
                       y - 0.025, y + 0.025,
                       (0.032, 0.060, 0.094, 1) if index % 2 == 0 else
                       (0.026, 0.050, 0.082, 1), dynamic=True)
            self._text(('  ' * depth) + track.label, self.ruler_left + 0.012,
                       y - 0.009, 0.024, (0.60, 0.74, 0.83, 1), dynamic=True)
            for interval in track.intervals:
                start, end = utc_seconds(interval.start), utc_seconds(interval.end)
                if end < bottom or start > top:
                    continue
                x0 = max(self.ruler_left, self._x_at(start))
                x1 = min(self.ruler_right, self._x_at(end))
                if x1 - x0 < 0.003:
                    x1 = min(self.ruler_right, x0 + 0.003)
                self._card('data interval', x0, x1, y - 0.020, y - 0.013,
                           interval.color, dynamic=True)
        self._playhead = self._card('UTC playhead', 0, 0.003,
                                    -0.95, self.top - 0.132,
                                    (0.99, 0.70, 0.28, 1), dynamic=True)
        self._update_playhead()

    def _update_playhead(self) -> None:
        x = self._x_at(self.playback.current)
        self._playhead.setX(x)
        self._playhead.show() if self.ruler_left <= x <= self.ruler_right else self._playhead.hide()

    def set_time(self, value: float) -> None:
        if self.follow and not self.interacting and value > self.center + self.span * 0.35:
            self.center = value + self.span * 0.15
            self._rebuild_ruler()
        self._readout.setText(datetime.fromtimestamp(value, timezone.utc).strftime('%Y-%m-%d  %H:%M:%S.%f')[:-3])
        self._play_text.setText('||' if self.playback.playing else '>')
        self._speed_text.setText(f'{self.playback.speed:g}x')
        self._update_playhead()

    @property
    def interacting(self) -> bool:
        return self._pointer_start is not None

    def contains(self, mouse) -> bool:
        return mouse.y <= self.top

    def begin_pointer(self, mouse) -> None:
        x = mouse.x * self.base.getAspectRatio()
        if x < self.ruler_left:
            if -0.89 <= mouse.y <= -0.75:
                for index, center in enumerate(self._button_centers):
                    if abs(x - center) <= 0.11:
                        if index == 0:
                            self.playback.change_speed(-1)
                        elif index == 1:
                            self.playback.toggle()
                        else:
                            self.playback.change_speed(1)
                        self.set_time(self.playback.current)
                        break
            return
        if x <= self.ruler_right:
            self._pointer_start = x
            self._pointer_center = self.center
            self._dragged = False

    def move_pointer(self, mouse) -> None:
        if not self.interacting:
            return
        x = mouse.x * self.base.getAspectRatio()
        distance = x - self._pointer_start
        if abs(distance) > 0.012:
            self._dragged = True
        if self._dragged:
            self.center = self._pointer_center - distance / self.ruler_width * self.span
            self.follow = False
            self._rebuild_ruler()

    def end_pointer(self, mouse=None) -> None:
        if self.interacting and not self._dragged and mouse is not None:
            x = mouse.x * self.base.getAspectRatio()
            if self.ruler_left <= x <= self.ruler_right:
                self.playback.seek(self._time_at(x))
                self.set_time(self.playback.current)
        self._pointer_start = None
        self._pointer_center = None

    def zoom(self, mouse, direction: int) -> None:
        x = max(self.ruler_left, min(self.ruler_right,
                                    mouse.x * self.base.getAspectRatio()))
        anchor = self._time_at(x)
        factor = self.config.zoom_factor ** direction
        self.span = max(self.config.min_span_seconds,
                        min(self.config.max_span_seconds, self.span * factor))
        fraction = (x - self.ruler_left) / self.ruler_width
        self.center = anchor + (0.5 - fraction) * self.span
        self.follow = False
        self._rebuild_ruler()

    def scroll_tracks(self, direction: int) -> None:
        """Scroll the visible track rows without changing the UTC viewport."""
        visible = max(1, int((self.config.height_fraction * 2 - 0.29) /
                             self.config.row_height))
        maximum = max(0, len(list(self.model.rows())) - visible)
        self.row_offset = max(0, min(maximum, self.row_offset + direction))
        self._rebuild_ruler()

    def refresh_tracks(self) -> None:
        """Redraw after callers add or edit intervals in the shared model."""
        self._rebuild_ruler()
