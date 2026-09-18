"""Generate the SSC26 poster QR codes.

Every code resolves to a page WE control under /ssc26/, never straight to
Colab or a PDF host. A printed QR is frozen forever; a URL we own can be
re-pointed after the poster is at the printer. That is also why the paper code
works before the proceedings PDF exists.

Distinct URLs per code so scans are attributable per QR in the analytics.

Error correction is set to H (~30% recoverable), which is what makes a code
survive a scuffed, curled or partly-shadowed poster at an angle across a
conference hall. Quiet zone is the spec-mandated 4 modules; printers that trim
it are the single most common cause of a code that will not scan.

Run:  python papers/SSC26_poster/generate_qr_codes.py
Out:  papers/SSC26_poster/output/qr_<name>.{png,svg}
"""

import os

import qrcode
import qrcode.image.svg
from qrcode.constants import ERROR_CORRECT_H

BASE = "https://nscheuer.github.io/Generalized_ADCS/ssc26"

# (filename stem, poster caption, url)
CODES = [
    ("paper",   "Full paper",              f"{BASE}/paper.html"),
    ("run",     "Runs in your browser",    f"{BASE}/run.html"),
    ("contact", "We help you port your law", f"{BASE}/contact.html"),
    # Not currently on the poster, but generated so it exists if a fourth code
    # is added or one of the above is repurposed.
    ("code",    "The code",                f"{BASE}/code.html"),
    ("landing", "SSC26 landing page",      f"{BASE}/"),
]

# 40 px per module at H correction gives a code that scans comfortably from
# ~1.5 m at roughly 4 cm printed, with headroom for the printer.
BOX_PIXELS = 40
BORDER_MODULES = 4


def build(url: str):
    qr = qrcode.QRCode(
        version=None,                  # smallest version that fits the payload
        error_correction=ERROR_CORRECT_H,
        box_size=BOX_PIXELS,
        border=BORDER_MODULES,
    )
    qr.add_data(url)
    qr.make(fit=True)
    return qr


def main() -> int:
    outdir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)

    print(f"{'file':<12} {'ver':>3} {'modules':>8} {'px':>6}  url")
    for stem, caption, url in CODES:
        qr = build(url)
        n_modules = qr.modules_count + 2 * BORDER_MODULES
        px = n_modules * BOX_PIXELS

        png_path = os.path.join(outdir, f"qr_{stem}.png")
        qr.make_image(fill_color="black", back_color="white").save(png_path)

        # Vector copy: a poster is typeset at print resolution, and an SVG
        # cannot be resampled into blur by the layout tool.
        svg = qrcode.QRCode(error_correction=ERROR_CORRECT_H,
                            border=BORDER_MODULES,
                            image_factory=qrcode.image.svg.SvgPathImage)
        svg.add_data(url)
        svg.make(fit=True)
        svg.make_image().save(os.path.join(outdir, f"qr_{stem}.svg"))

        print(f"qr_{stem:<9} {qr.version:>3} {n_modules:>8} {px:>6}  {url}")

    print(f"\nwrote {2 * len(CODES)} files to {outdir}")
    print("\nPrint checklist:")
    print("  - keep the white quiet zone; do not crop to the black edge")
    print("  - >= 3 cm printed for a poster read at arm's length; 4 cm is safer")
    print("  - do not recolour or place on a busy background: contrast is what scans")
    print("  - scan every code off the actual printed proof, not off the screen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
