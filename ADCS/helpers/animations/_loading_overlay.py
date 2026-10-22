"""Small centered loading indicator for blocking graphics setup."""


class LoadingOverlay:
    def __init__(self, base):
        from direct.gui.DirectGui import DirectFrame, DirectLabel

        self.base = base
        self.panel = DirectFrame(
            parent=base.aspect2d, frameSize=(-0.55, 0.55, -0.11, 0.11),
            frameColor=(0.035, 0.055, 0.09, 0.96), relief=None,
            sortOrder=100,
        )
        self.label = DirectLabel(
            parent=self.panel, text='Loading graphics', pos=(0, 0, 0.045),
            text_scale=0.04, text_fg=(0.91, 0.95, 1.0, 1),
            frameColor=(0, 0, 0, 0),
        )
        self.track = DirectFrame(
            parent=self.panel, frameSize=(-0.49, 0.49, -0.045, -0.012),
            frameColor=(0.13, 0.18, 0.24, 1), relief=None,
        )
        self.fill = DirectFrame(
            parent=self.panel, frameSize=(-0.49, -0.49, -0.045, -0.012),
            frameColor=(0.28, 0.70, 1.0, 1), relief=None,
        )
        self.update(0.0, 'Loading graphics')

    def update(self, fraction, label):
        fraction = max(0.0, min(1.0, float(fraction)))
        self.fill['frameSize'] = (-0.49, -0.49 + 0.98 * fraction, -0.045, -0.012)
        self.label['text'] = label
        self.base.graphicsEngine.renderFrame()

    def close(self):
        self.panel.destroy()
