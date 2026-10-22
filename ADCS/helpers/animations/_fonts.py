"""Bundled, portable UI typeface."""

from pathlib import Path


def load_ui_font(base):
    """Load Liberation Sans, a clean Arial-like font bundled with the viewer."""
    path = Path(__file__).with_name('assets') / 'LiberationSans-Regular.ttf'
    font = base.loader.loadFont(str(path))
    if font is None:
        raise RuntimeError(f'Could not load viewer font: {path}')
    font.setPixelsPerUnit(64)
    return font
