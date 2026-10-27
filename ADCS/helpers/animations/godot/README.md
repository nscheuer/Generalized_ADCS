# Godot orbit viewer

Run the example with the repository venv:

```bash
.venv/bin/python debug/debug_helpers/debug_animations/debug_orbit_godot_simulation.py
```

Python callers can pass a `RunResults` or `SimulationResults` to
`ADCS.helpers.animations.animate_orbit_godot(results=...)`, or the three
histories accepted by the Panda3D entry point. `output=` keeps the generated
vizlog; otherwise it is deleted after the window closes. Low graphics is the
default. On first run with missing planet textures, the viewer asks in the
terminal whether to download 2K maps and then optional 8K maps. The Godot
window displays a notice to check the terminal while those prompts are active.
Downloads use `AssetManager`; declining leaves the viewer running with built-in
colors and any maps already cached.

The Linux x86_64 Godot 4.7.2 executable is bundled as a verified ZIP in the
wheel. On first use it is unpacked into `ADCS/helpers/animations/assets/bin`.
For other platforms, pass `binary=` or set `ADCS_VIEWER_BIN` until their
bundled builds are added.

Optional textures and atmosphere data are placed under
`ADCS/helpers/animations/assets`. They are ignored by Git and excluded from
the wheel. Prepare them explicitly with:

```bash
.venv/bin/python -m ADCS.helpers.animations.godot.assets list
.venv/bin/python -m ADCS.helpers.animations.godot.assets fetch <source-id>
.venv/bin/python -m ADCS.helpers.animations.godot.assets fetch-planets --lod 2k
.venv/bin/python -m ADCS.helpers.animations.godot.assets fetch-planets --lod 8k
.venv/bin/python -m ADCS.helpers.animations.godot.assets build
```

Each missing raw asset is resolved through `AssetManager`, which asks before
downloading. Scene assets are optional: without them the viewer uses flat
colors and procedural stars. Spacecraft models use a `.viz.json` sidecar whose
LOD entries reference `.glb` files.

Planet maps, including Earth's day, night, cloud, normal, and specular maps, are stored
in `ADCS/helpers/animations/assets/planet_textures`. Low graphics uses 2K maps;
high graphics switches to the locally available 8K maps when a body's projected
radius is at least 128 pixels. Solar System Scope offers Venus at 4K and Uranus
and Neptune at 2K, so those use the closest available resolution.

The Earth atmosphere is enabled without an asset download. It uses exponential
Rayleigh and Mie density, scattering, and aerial perspective following NVIDIA's
GPU Gems 2 atmospheric scattering chapter. If locally built atmosphere tables
are present, the viewer uses their higher accuracy transmittance and multiple
scattering data instead.
