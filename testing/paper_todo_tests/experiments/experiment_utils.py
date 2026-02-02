"""
Experiment Utilities
====================

Common utilities for paper experiment scripts:
- Publication-quality figure setup (TeX fonts, clean styling)
- Simulation infrastructure
- Data saving/loading
- Progress reporting
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import json
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.disturbances import GG_Disturbance
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import normalize, quat_mult, quat_inv, rot_mat, skewsym
from ADCS.helpers.math_constants import MathConstants
from ADCS.CONOPS.goals import Goal, ReducedAttitudeGoal, FullAttitudeGoal


# =============================================================================
# FIGURE SETUP
# =============================================================================

def setup_publication_style():
    """
    Configure matplotlib for publication-quality figures.
    
    Uses Computer Modern (TeX) fonts and clean styling.
    """
    # Check if TeX is available
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    except:
        # Fallback if TeX not available
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'cm'
    
    # Clean styling
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.05
    
    # Font sizes
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    
    # Lines and markers
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['lines.markersize'] = 4
    
    # Axes
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Legend
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'best'
    
    # Ticks
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4


# Publication color palette (colorblind-friendly)
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'gray': '#999999',
    'black': '#000000',
}


def save_figure(fig: plt.Figure, path: Path, formats: List[str] = ['png', 'pdf']):
    """Save figure in multiple formats."""
    path = Path(path)
    for fmt in formats:
        fig.savefig(path.with_suffix(f'.{fmt}'), format=fmt)
    print(f"  Saved: {path.stem}")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SimConfig:
    """Simulation configuration."""
    duration_s: float = 500.0
    dt_s: float = 1.0
    altitude_km: float = 400.0
    inclination_deg: float = 51.6
    seed: Optional[int] = None


@dataclass  
class SimState:
    """Spacecraft state at a single timestep."""
    t: float
    omega: np.ndarray  # Angular velocity [rad/s]
    q: np.ndarray      # Quaternion [w, x, y, z]
    h_rw: np.ndarray   # RW momentum if applicable
    
    def to_array(self) -> np.ndarray:
        """Convert to flat array for integration."""
        return np.hstack([self.omega, self.q, self.h_rw])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, t: float = 0.0) -> 'SimState':
        """Create from flat array."""
        n_rw = max(0, len(arr) - 7)
        return cls(
            t=t,
            omega=arr[0:3],
            q=arr[3:7],
            h_rw=arr[7:7+n_rw] if n_rw > 0 else np.array([])
        )


@dataclass
class TrialResult:
    """Result from a single simulation trial."""
    trial_id: int
    config_name: str
    
    # Error metrics
    final_error_deg: float
    mean_error_deg: float
    max_error_deg: float
    
    # Time series (optional, for plotting)
    times: Optional[np.ndarray] = None
    errors_deg: Optional[np.ndarray] = None
    
    # Convergence
    converged: bool = True
    settling_time_s: Optional[float] = None
    
    # Initial conditions (for reproducibility)
    q_init: Optional[np.ndarray] = None
    q_goal: Optional[np.ndarray] = None


@dataclass
class CampaignResult:
    """Aggregate results from Monte Carlo campaign."""
    config_name: str
    n_trials: int
    
    # Statistics
    mean_final_error_deg: float
    std_final_error_deg: float
    median_final_error_deg: float
    
    # Success rates
    pct_within_1deg: float
    pct_within_5deg: float
    pct_within_10deg: float
    
    # All errors for plotting
    all_errors_deg: List[float] = field(default_factory=list)


# =============================================================================
# QUATERNION UTILITIES
# =============================================================================

def random_quaternion() -> np.ndarray:
    """Generate uniformly random quaternion over SO(3)."""
    u = np.random.random(3)
    q = np.array([
        np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
        np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
        np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
        np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
    ])
    # Convert to [w, x, y, z] convention
    return np.array([q[3], q[0], q[1], q[2]])


def quaternion_error_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute angle between two quaternions in degrees."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # q_err = q1^{-1} * q2
    q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    q_err = quat_mult(q1_inv, q2)
    
    # Angle from scalar part
    angle_rad = 2 * np.arccos(np.clip(abs(q_err[0]), 0, 1))
    return np.rad2deg(angle_rad)


def vector_pointing_error_deg(q: np.ndarray, body_vec: np.ndarray, target_vec_eci: np.ndarray) -> float:
    """
    Compute pointing error for reduced-attitude (vector alignment) goal.
    
    Args:
        q: Current quaternion [w, x, y, z]
        body_vec: Body-frame vector to align (e.g., [0, 1, 0] for y-axis)
        target_vec_eci: Target direction in ECI frame
    
    Returns:
        Angle between rotated body vector and target (degrees)
    """
    R = rot_mat(q)
    body_vec_eci = R @ body_vec
    
    body_vec_eci = body_vec_eci / np.linalg.norm(body_vec_eci)
    target_vec_eci = target_vec_eci / np.linalg.norm(target_vec_eci)
    
    cos_angle = np.clip(np.dot(body_vec_eci, target_vec_eci), -1, 1)
    return np.rad2deg(np.arccos(cos_angle))


# =============================================================================
# ORBIT UTILITIES
# =============================================================================

def create_orbit(config: SimConfig, t0: float = 0.0) -> Orbit:
    """Create orbit object for simulation."""
    ephem = Ephemeris()
    
    # Convert altitude to position
    r_mag = 6371 + config.altitude_km  # km
    inc_rad = np.deg2rad(config.inclination_deg)
    
    # Initial position (ascending node)
    R = np.array([r_mag, 0, 0])
    
    # Circular orbit velocity
    mu = 398600.4418  # km³/s²
    v_mag = np.sqrt(mu / r_mag)
    V = np.array([0, v_mag * np.cos(inc_rad), v_mag * np.sin(inc_rad)])
    
    # Time span in Julian centuries
    start_j2000 = 0.22
    end_j2000 = start_j2000 + config.duration_s * TimeConstants.sec2cent
    
    os = Orbital_State(ephem=ephem, J2000=start_j2000, R=R, V=V)
    orb = Orbit(os0=os, end_time=end_j2000, dt=config.dt_s, use_J2=True, fast=False)
    
    return orb


def get_orbital_state(orb: Orbit, t: float, t0_j2000: float = 0.22) -> Orbital_State:
    """Get orbital state at time t seconds from simulation start."""
    j2000 = t0_j2000 + t * TimeConstants.sec2cent
    return orb.get_os(j2000)


# =============================================================================
# DATA I/O
# =============================================================================

def save_results(results: Dict[str, Any], path: Path):
    """Save results to JSON file."""
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, '__dict__'):
            return {k: convert(v) for k, v in obj.__dict__.items()}
        return obj
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)


def load_results(path: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_latex_table(
    results: Dict[str, CampaignResult],
    caption: str,
    label: str
) -> str:
    """Generate LaTeX table from campaign results."""
    
    latex = r"""\begin{table}[htbp]
    \centering
    \caption{""" + caption + r"""}
    \label{""" + label + r"""}
    \begin{tabular}{lccccc}
        \toprule
        Configuration & Mean & Std & $<1^\circ$ & $<5^\circ$ & $<10^\circ$ \\
        \midrule
"""
    
    for name, res in results.items():
        latex += f"        {name} & {res.mean_final_error_deg:.2f}$^\\circ$ & "
        latex += f"{res.std_final_error_deg:.2f}$^\\circ$ & "
        latex += f"{res.pct_within_1deg:.0f}\\% & "
        latex += f"{res.pct_within_5deg:.0f}\\% & "
        latex += f"{res.pct_within_10deg:.0f}\\% \\\\\n"
    
    latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
    return latex


# =============================================================================
# PROGRESS AND REPORTING
# =============================================================================

def print_header(text: str):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)


def print_subheader(text: str):
    """Print formatted subheader."""
    print(f"\n  -- {text} --")


def print_results_table(results: Dict[str, CampaignResult]):
    """Print results as formatted table."""
    print(f"\n  {'Configuration':<25} {'Mean':>8} {'Std':>8} {'<1°':>6} {'<5°':>6} {'<10°':>6}")
    print("  " + "-"*65)
    for name, res in results.items():
        print(f"  {name:<25} {res.mean_final_error_deg:>7.2f}° {res.std_final_error_deg:>7.2f}° "
              f"{res.pct_within_1deg:>5.0f}% {res.pct_within_5deg:>5.0f}% {res.pct_within_10deg:>5.0f}%")


# Initialize matplotlib style on import
setup_publication_style()
