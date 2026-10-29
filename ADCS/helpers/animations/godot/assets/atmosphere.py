r"""
Precomputed atmosphere tables for the viewer, after Hillaire (2020), "A
Scalable and Production Ready Sky and Atmosphere Rendering Technique" (CGF
39(4); reference code: github.com/sebh/UnrealEngineSkyAtmosphere, MIT).

Two small tables are computed here; the viewer ray-marches the atmosphere per
pixel with them (the method's path for views from space):

* **transmittance** ``T(r, mu)`` from a point at radius ``r`` along a ray
  with zenith cosine ``mu`` to the top of the atmosphere, using the texture
  parameterisation of Bruneton (2017), so it stays accurate near the horizon;
* **multiple scattering** ``Psi(r, mu_s)``: radiance from all orders >= 2 per
  unit scattering coefficient, for unit solar irradiance (Hillaire's
  infinite-order approximation, isotropic phase, ground albedo bounce).

Lengths are metres. The atmosphere sits on Earth's equatorial radius so it
matches the rendered globe. Rayleigh, Mie and ozone coefficients follow
Bruneton and Hillaire; scale heights are slightly broadened for a more visible
orbital limb.
"""

__all__ = ["AtmosphereParams", "transmittance_table", "multiscattering_table", "transmittance_uv", "build_tables"]

from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class AtmosphereParams:
    bottom_radius: float = 6378137.0
    top_radius: float = 6378137.0 + 120.0e3
    rayleigh_scattering: Tuple[float, float, float] = (5.802e-6, 13.558e-6, 33.100e-6)  # 1/m at 680, 550, 440 nm
    rayleigh_scale_height: float = 9500.0
    mie_scattering: Tuple[float, float, float] = (3.996e-6, 3.996e-6, 3.996e-6)
    mie_extinction: Tuple[float, float, float] = (4.440e-6, 4.440e-6, 4.440e-6)
    mie_scale_height: float = 1450.0
    mie_g: float = 0.8
    ozone_absorption: Tuple[float, float, float] = (0.650e-6, 1.881e-6, 0.085e-6)
    ozone_center: float = 25.0e3
    ozone_half_width: float = 15.0e3
    ground_albedo: float = 0.3
    transmittance_size: Tuple[int, int] = field(default=(256, 64))   # (mu, r)
    multiscattering_size: Tuple[int, int] = field(default=(32, 32))  # (mu_s, r)


def _densities(p: AtmosphereParams, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho_r = np.exp(-np.maximum(h, 0.0) / p.rayleigh_scale_height)
    rho_m = np.exp(-np.maximum(h, 0.0) / p.mie_scale_height)
    rho_o = np.maximum(0.0, 1.0 - np.abs(h - p.ozone_center) / p.ozone_half_width)
    return rho_r, rho_m, rho_o


def _extinction(p: AtmosphereParams, h: np.ndarray) -> np.ndarray:
    rho_r, rho_m, rho_o = _densities(p, h)
    return (rho_r[..., None] * np.asarray(p.rayleigh_scattering) + rho_m[..., None] * np.asarray(p.mie_extinction)
            + rho_o[..., None] * np.asarray(p.ozone_absorption))


def _distance_to_top(p: AtmosphereParams, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
    disc = r * r * (mu * mu - 1.0) + p.top_radius**2
    return np.maximum(-r * mu + np.sqrt(np.maximum(disc, 0.0)), 0.0)


def _unit_from_coord(u: np.ndarray, n: int) -> np.ndarray:
    return (u - 0.5 / n) / (1.0 - 1.0 / n)


def _coord_from_unit(x: np.ndarray, n: int) -> np.ndarray:
    return 0.5 / n + x * (1.0 - 1.0 / n)


def transmittance_uv(p: AtmosphereParams, r: np.ndarray, mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Texture coordinates of (r, mu) in the transmittance table (Bruneton 2017)."""
    W, Hn = p.transmittance_size
    H = np.sqrt(p.top_radius**2 - p.bottom_radius**2)
    rho = np.sqrt(np.maximum(r * r - p.bottom_radius**2, 0.0))
    d = _distance_to_top(p, r, mu)
    d_min = p.top_radius - r
    d_max = rho + H
    x_mu = (d - d_min) / np.maximum(d_max - d_min, 1e-9)
    return _coord_from_unit(x_mu, W), _coord_from_unit(rho / H, Hn)


def transmittance_table(p: AtmosphereParams, steps: int = 500) -> np.ndarray:
    r"""``(H, W, 3)`` float32 table; rows are r (bottom to top), columns mu."""
    W, Hn = p.transmittance_size
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(Hn) + 0.5) / Hn
    x_mu, x_r = np.meshgrid(_unit_from_coord(u, W), _unit_from_coord(v, Hn))
    H = np.sqrt(p.top_radius**2 - p.bottom_radius**2)
    rho = H * x_r
    r = np.sqrt(rho * rho + p.bottom_radius**2)
    d_min = p.top_radius - r
    d_max = rho + H
    d = d_min + x_mu * (d_max - d_min)
    mu = np.where(d == 0.0, 1.0, (H * H - rho * rho - d * d) / np.maximum(2.0 * r * d, 1e-9))
    mu = np.clip(mu, -1.0, 1.0)
    # Optical depth by the trapezoidal rule along the ray to the top.
    t = np.linspace(0.0, 1.0, steps + 1)[None, None, :] * d[..., None]
    ri = np.sqrt(t * t + 2.0 * r[..., None] * mu[..., None] * t + (r * r)[..., None])
    ext = _extinction(p, ri - p.bottom_radius)
    depth = np.trapezoid(ext, t[..., None], axis=2)
    return np.exp(-depth).astype(np.float32)


def _sample_transmittance(p: AtmosphereParams, table: np.ndarray, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Bilinear lookup of the transmittance table (numpy mirror of the shader)."""
    Hn, W = table.shape[:2]
    u, v = transmittance_uv(p, r, mu)
    x = np.clip(u * W - 0.5, 0, W - 1)
    y = np.clip(v * Hn - 0.5, 0, Hn - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, Hn - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    top = table[y0, x0] * (1 - fx) + table[y0, x1] * fx
    bot = table[y1, x0] * (1 - fx) + table[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def _ray_sphere(r: np.ndarray, mu: np.ndarray, radius: float) -> np.ndarray:
    """Distance along the ray to a sphere of `radius` (first hit), NaN if missed."""
    disc = r * r * (mu * mu - 1.0) + radius * radius
    with np.errstate(invalid="ignore"):
        t = -r * mu - np.sqrt(disc)
    return np.where((disc >= 0.0) & (t > 0.0), t, np.nan)


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + 5.0**0.5) * i
    s = np.sqrt(1.0 - z * z)
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=-1)


def multiscattering_table(p: AtmosphereParams, transmittance: np.ndarray, directions: int = 64,
                          steps: int = 40) -> np.ndarray:
    r"""
    ``(H, W, 3)`` float32 table of Psi_ms; rows are altitude (bottom to top),
    columns mu_s in [-1, 1] (Hillaire 2020, section 5.5).

    For every texel, rays are marched in ``directions`` uniformly spread
    directions; ``L2`` is second-order radiance from single-scattered sunlight
    (isotropic phase) plus the ground bounce, ``f_ms`` the scattered fraction of
    a unit isotropic illumination. Psi = L2 / (1 - f_ms) sums all orders.
    """
    W, Hn = p.multiscattering_size
    mu_s = -1.0 + 2.0 * (np.arange(W) + 0.5) / W
    alt = (np.arange(Hn) + 0.5) / Hn
    r0 = p.bottom_radius + alt * (p.top_radius - p.bottom_radius)
    dirs = _fibonacci_sphere(directions)
    sigma_r = np.asarray(p.rayleigh_scattering)
    sigma_m = np.asarray(p.mie_scattering)
    iso = 1.0 / (4.0 * np.pi)
    out = np.zeros((Hn, W, 3))
    for j, r in enumerate(r0):
        x0 = np.array([0.0, 0.0, r])
        for i, ms in enumerate(mu_s):
            sun = np.array([np.sqrt(1.0 - ms * ms), 0.0, ms])
            mu = dirs[:, 2]
            t_ground = _ray_sphere(np.full(len(dirs), r), mu, p.bottom_radius)
            t_top = _distance_to_top(p, np.full(len(dirs), r), mu)
            t_max = np.where(np.isnan(t_ground), t_top, t_ground)
            dt = t_max / steps
            L = np.zeros((len(dirs), 3))
            fms = np.zeros((len(dirs), 3))
            T = np.ones((len(dirs), 3))
            for k in range(steps):
                t = (k + 0.5) * dt
                x = x0[None, :] + dirs * t[:, None]
                rx = np.linalg.norm(x, axis=1)
                h = rx - p.bottom_radius
                rho_r, rho_m, _ = _densities(p, h)
                ss = rho_r[:, None] * sigma_r + rho_m[:, None] * sigma_m
                st = _extinction(p, h)
                mus_x = (x @ sun) / rx
                # Sun hidden by the Earth from this point?
                lit = mus_x > -np.sqrt(np.maximum(1.0 - (p.bottom_radius / rx) ** 2, 0.0))
                Ts = _sample_transmittance(p, transmittance, rx, mus_x) * lit[:, None]
                step_T = np.exp(-st * dt[:, None])
                integ = (1.0 - step_T) / np.maximum(st, 1e-30)
                L += T * ss * Ts * iso * integ
                fms += T * ss * integ
                T *= step_T
            hit = ~np.isnan(t_ground)
            if hit.any():
                xg = x0[None, :] + dirs[hit] * t_ground[hit, None]
                rg = np.linalg.norm(xg, axis=1)
                mug = np.clip((xg @ sun) / rg, 0.0, 1.0)
                Tg = _sample_transmittance(p, transmittance, rg, mug)
                L[hit] += T[hit] * Tg * mug[:, None] * p.ground_albedo / np.pi
            # Averages over the sphere: sum x solid angle (4 pi / N) x isotropic phase (1 / 4 pi).
            l2 = L.mean(axis=0)
            f = fms.mean(axis=0)
            out[j, i] = l2 / (1.0 - np.minimum(f, 0.99))
    return out.astype(np.float32)


def build_tables(p: AtmosphereParams = AtmosphereParams()) -> Dict[str, np.ndarray]:
    t = transmittance_table(p)
    return {"transmittance": t, "multiscattering": multiscattering_table(p, t)}


def params_json(p: AtmosphereParams) -> dict:
    d = asdict(p)
    d["reference"] = "Hillaire 2020 (CGF 39(4)); transmittance parameterisation Bruneton 2017"
    d["units"] = "metres, 1/metres; Psi_ms per unit solar irradiance and unit scattering coefficient"
    return d
